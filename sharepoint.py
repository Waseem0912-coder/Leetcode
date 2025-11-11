# rag_report_generator.py

import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import pypdf
import faiss
import numpy as np
from docx import Document
import warnings

# Suppress a specific warning from the pypdf library
warnings.filterwarnings("ignore", category=pypdf.errors.PdfReadWarning)

# --- Configuration ---
PDF_FOLDER = "pdf"
EMBEDDING_MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
LLM_NAME = "Qwen/Qwen3-1.7B"
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 150 # Number of characters to overlap between chunks
TOP_K_RESULTS = 5 # Number of relevant chunks to retrieve for each topic
OUTPUT_FILENAME = "RAG_Generated_Report.docx"

# --- Helper Functions for Embedding (from your example) ---
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# --- Core RAG Application Functions ---

def load_models():
    """Loads the embedding and language models with quantization/lower precision."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    # Using bfloat16 for memory efficiency
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, padding_side='left')
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    embedding_model.eval()
    print("Embedding model loaded.")

    # 2. Load Language Model (LLM) for generation
    print(f"Loading language model: {LLM_NAME}...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype="auto", # Automatically uses bfloat16 on compatible hardware
        device_map="auto"   # Automatically handles device placement
    )
    print("Language model loaded.")
    
    return embedding_model, embedding_tokenizer, llm, llm_tokenizer, device

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks."""
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

def create_embeddings(texts, model, tokenizer, device, batch_size=32):
    """Creates embeddings for a list of text chunks in batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()

def build_vector_store(embeddings):
    """Builds a FAISS index for efficient similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_vector_store(query_embedding, index, k=TOP_K_RESULTS):
    """Searches the vector store for the top_k most similar chunks."""
    distances, indices = index.search(query_embedding, k)
    return indices[0]

def generate_llm_response(prompt, model, tokenizer):
    """Generates a response from the LLM using a given prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024, # Limit response length
        do_sample=False
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return content

# --- Main Execution Logic ---
if __name__ == "__main__":
    # 1. Setup
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created folder '{PDF_FOLDER}'. Please add your PDF files there and run again.")
        exit()

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in the '{PDF_FOLDER}' directory. Please add some PDFs.")
        exit()
        
    print(f"Found {len(pdf_files)} PDF(s) to process: {', '.join(pdf_files)}")

    # 2. Load Models
    embed_model, embed_tok, llm, llm_tok, device = load_models()
    
    # Create a new Word document
    doc = Document()
    doc.add_heading('RAG-Based PDF Analysis Report', level=0)

    # 3. Process each PDF
    for pdf_file in pdf_files:
        print(f"\n{'='*50}\nProcessing: {pdf_file}\n{'='*50}")
        doc.add_heading(f"Report for: {pdf_file}", level=1)
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)

        # a. Extract and Chunk Text
        print("Step 1: Extracting and chunking text...")
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text or not full_text.strip():
            print(f"Could not extract text from {pdf_file}, or it is empty. Skipping.")
            doc.add_paragraph("Could not extract text from this document.")
            doc.add_page_break()
            continue
        chunks = chunk_text(full_text)
        print(f"Created {len(chunks)} text chunks.")

        # b. Create Embeddings and Vector Store
        print("Step 2: Creating embeddings and building vector store...")
        chunk_embeddings = create_embeddings(chunks, embed_model, embed_tok, device)
        vector_store = build_vector_store(chunk_embeddings)
        print("Vector store created.")

        # c. Identify Topics in the Document using the LLM
        print("Step 3: Identifying main topics...")
        topic_identification_prompt = f"""
        Analyze the beginning of the following document and identify the main sections or key topics discussed.
        List them as a comma-separated list. For example: 'Introduction, Methodology, Key Findings, Conclusion'.
        
        Document Start:
        "{' '.join(chunks[:2])[:2000]}" 
        """
        topic_list_str = generate_llm_response(topic_identification_prompt, llm, llm_tok)
        topics = [topic.strip() for topic in topic_list_str.split(',') if topic.strip()]
        print(f"Identified Topics: {topics}")
        if not topics:
            topics = ["General Summary"] # Fallback topic

        # d. Generate Report Section by Section
        print("Step 4: Generating report for each topic...")
        for topic in topics:
            print(f"  - Generating summary for topic: '{topic}'")
            doc.add_heading(topic, level=2)

            # Find relevant chunks for the current topic
            query_embedding = create_embeddings([topic], embed_model, embed_tok, device)
            relevant_indices = search_vector_store(query_embedding, vector_store)
            context = "\n\n---\n\n".join([chunks[i] for i in relevant_indices])

            # Ask the LLM to summarize the findings for this topic based on context
            summary_prompt = f"""
            You are a professional analyst. Based ONLY on the provided context below, write a detailed summary about the topic: '{topic}'.
            Focus on organizing all the unique things that happened, key data points, or conclusions mentioned.
            Do not invent information. Structure your answer as a clear and concise report section.
            
            CONTEXT:
            {context}
            """
            section_report = generate_llm_response(summary_prompt, llm, llm_tok)
            doc.add_paragraph(section_report)

        doc.add_page_break()

    # 4. Save the final document
    doc.save(OUTPUT_FILENAME)
    print(f"\n{'='*50}\nReport generation complete.\nOutput saved to: {OUTPUT_FILENAME}\n{'='*50}")
