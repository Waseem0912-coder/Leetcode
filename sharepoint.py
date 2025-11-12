#!/usr/bin/env python

import os
import re
import json
import logging
from typing import List

import torch
import torch.nn.functional as F
from docx import Document

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from chromadb.config import Settings

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("pypdf").setLevel(logging.ERROR)

PDF_SOURCE_DIR = "./source_pdfs"
VECTORSTORE_DIR = "./chroma_db_qwen"
OUTPUT_DOCX_FILE = "Consolidated_Report.docx"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    logging.warning("CUDA not available. The process will be very slow on CPU.")

# --- Helper Function for Cleaning LLM Output ---
def clean_llm_output(text: str) -> str:
    """
    Uses regex to remove <think>...</think> blocks and trims whitespace.
    The JsonOutputParser will handle extracting the actual JSON.
    """
    # Remove the <think> blocks, as they can confuse the JSON parser
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# --- Step 1: Custom Qwen3 Embedding Class (No changes here) ---
class CustomQwenEmbeddings(Embeddings):
    """Custom LangChain embedding class for Qwen/Qwen3-Embedding-0.6B."""
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device
        self.task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        logging.info(f"CustomQwenEmbeddings initialized with model '{model_name}' on device '{device}'.")

    def _get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task_description}\nQuery: {query}'

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        max_length = 4096
        inputs = self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
        
        embeddings = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.to(torch.float32).cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        instructed_query = self._get_detailed_instruct(text)
        max_length = 4096
        inputs = self.tokenizer([instructed_query], max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        embedding = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return normalized_embedding.to(torch.float32).cpu().tolist()[0]

def setup_directories():
    """Ensures the source PDF directory exists."""
    if not os.path.exists(PDF_SOURCE_DIR):
        os.makedirs(PDF_SOURCE_DIR)
        logging.info(f"Created directory: {PDF_SOURCE_DIR}")
        logging.warning(f"Please add your PDF files to the '{PDF_SOURCE_DIR}' directory before running again.")
        exit()

def load_and_index_documents() -> Chroma:
    """Step 2: Handles the ingestion and indexing of PDF documents."""
    logging.info("--- Step 2: Document Ingestion & Indexing ---")
    
    chroma_settings = Settings(anonymized_telemetry=False)

    if os.path.exists(VECTORSTORE_DIR):
        logging.info(f"Loading existing vector store from {VECTORSTORE_DIR}...")
        embedding_function = CustomQwenEmbeddings()
        vector_store = Chroma(
            persist_directory=VECTORSTORE_DIR, 
            embedding_function=embedding_function,
            client_settings=chroma_settings
        )
        logging.info("Vector store loaded successfully.")
        return vector_store

    logging.info(f"No existing vector store found. Starting ingestion from {PDF_SOURCE_DIR}...")
    loader = PyPDFDirectoryLoader(PDF_SOURCE_DIR)
    documents = loader.load()

    if not documents:
        logging.error(f"No PDF documents found in '{PDF_SOURCE_DIR}'. Please add files and restart.")
        exit()
    
    logging.info(f"Loaded {len(documents)} document pages.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")

    logging.info("Initializing custom embedding model for indexing...")
    embedding_function = CustomQwenEmbeddings()

    logging.info("Creating and persisting vector store... This may take a while.")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=VECTORSTORE_DIR,
        client_settings=chroma_settings
    )
    logging.info(f"Vector store created and saved to {VECTORSTORE_DIR}.")
    return vector_store

def load_generator_llm() -> HuggingFacePipeline:
    """Step 3: Loads the Qwen3-1.7B generator model with 4-bit quantization."""
    logging.info("--- Step 3: Loading Generator LLM ---")
    
    model_id = "Qwen/Qwen3-1.7B"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    logging.info(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logging.info(f"Loading model '{model_id}' with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        temperature=0.0,
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    logging.info("Generator LLM loaded and wrapped in LangChain pipeline.")
    return llm

def generate_dynamic_topics(vector_store: Chroma, llm: HuggingFacePipeline) -> List[str]:
    """Step 4: Dynamically generates a list of main topics from the documents."""
    logging.info("--- Step 4: Dynamic Topic Generation (Pass 1) ---")
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    
    # --- FIXED: Ensures .invoke() is used ---
    sample_docs = retriever.invoke("What are the main subjects, themes, and topics in these documents?")
    context_text = "\n\n".join([doc.page_content for doc in sample_docs])

    prompt_template = """<|system|>
You are an expert analyst. Your task is to identify the main topics from the provided text.
Base your answer ONLY on the provided context. Respond ONLY with a JSON list of strings.
If no topics can be identified, respond with an empty list [].<|user|>
Context:
{context}

Analyze the context above. Identify the 5-7 most important, high-level topics discussed.
Do not invent topics not present in the text.
Your response must be a single JSON list of strings.
Example: ["Project Budget Analysis", "Security Protocol Review", "Quarterly Performance Metrics"]<|assistant|>
"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    # --- FIXED: Uses the corrected clean_llm_output function ---
    topic_chain = prompt | llm | RunnableLambda(clean_llm_output) | JsonOutputParser()

    logging.info("Invoking LLM to generate topics...")
    try:
        topics = topic_chain.invoke({"context": context_text})
        logging.info(f"Dynamically generated topics: {topics}")
        if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
            raise ValueError("LLM did not return a valid JSON list of strings for topics.")
        return topics
    except Exception as e:
        logging.error(f"Failed to generate topics: {e}. Using fallback topics.")
        return ["General Analysis", "Key Findings", "Recommendations"]

import torch # Add this import at the top of your script

def process_topics_and_extract_data(topics: List[str], vector_store: Chroma, llm: HuggingFacePipeline) -> dict:
    """Step 5: Loops through topics, retrieves relevant context, and extracts key points."""
    logging.info("--- Step 5: Topic-Based Extraction Loop (Pass 2 & 3) ---")
    compiled_data = {}
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    extraction_prompt_template = """<|system|>
You are a meticulous data extraction assistant. Your sole purpose is to extract key facts and bullet points from the given context that are directly related to the specified topic.
You must base your answer ONLY on the provided context. Do not add any information that is not present.
The context may contain unstructured text extracted from tables; interpret it as best as you can to pull out key data points.
Your response must be ONLY a JSON list of strings.
If no relevant information is found in the context for the given topic, you must respond with an empty list [].<|user|>
Context:
{context}

Topic: "{topic}"

Based ONLY on the context provided above, extract all key bullet points, facts, and data points related to the topic.<|assistant|>
"""
    extraction_prompt = PromptTemplate(template=extraction_prompt_template, input_variables=["context", "topic"])
    extraction_chain = extraction_prompt | llm | RunnableLambda(clean_llm_output) | JsonOutputParser()

    for i, topic in enumerate(topics):
        logging.info(f"Processing topic {i+1}/{len(topics)}: '{topic}'")
        
        retrieved_docs = retriever.invoke(topic)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        try:
            extracted_points = extraction_chain.invoke({"context": context, "topic": topic})
            if not isinstance(extracted_points, list):
                logging.warning(f"LLM returned non-list data for topic '{topic}'. Skipping.")
                extracted_points = []
        except Exception as e:
            logging.error(f"An error occurred during extraction for topic '{topic}': {e}")
            extracted_points = []
        finally:
            # Explicitly clear CUDA cache to prevent memory leaks in the loop
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        unique_points = sorted(list(set(point.strip() for point in extracted_points if isinstance(point, str) and point.strip())))
        
        if unique_points:
            logging.info(f"  - Extracted and deduplicated {len(unique_points)} points for '{topic}'.")
            compiled_data[topic] = unique_points
        else:
            logging.info(f"  - No relevant points found for '{topic}'.")

    return compiled_data

def generate_docx_report(data: dict, filename: str):
    """Step 6: Generates a .docx report from the compiled data."""
    logging.info(f"--- Step 6: Generating DOCX Report ---")
    doc = Document()
    doc.add_heading('Consolidated RAG Analysis Report', level=0)
    doc.add_paragraph(f'This report was automatically generated by analyzing documents in the "{PDF_SOURCE_DIR}" directory.')
    doc.add_paragraph()

    if not data:
        doc.add_paragraph("No data was successfully extracted from the source documents.")
    else:
        for topic, points in data.items():
            doc.add_heading(topic, level=1)
            if not points:
                doc.add_paragraph("No specific bullet points were extracted for this topic.")
            else:
                for point in points:
                    # Clean any lingering <think> tags just in case
                    clean_point = re.sub(r"<think>.*?</think>", "", point, flags=re.DOTALL).strip()
                    if clean_point:
                        doc.add_paragraph(clean_point, style='List Bullet')
            doc.add_paragraph()

    doc.save(filename)
    logging.info(f"Report successfully saved as '{filename}'")

def main():
    """Main function to orchestrate the entire RAG pipeline."""
    logging.info("Starting the Automated Report Generation Pipeline.")
    
    setup_directories()
    vector_store = load_and_index_documents()
    llm = load_generator_llm()
    dynamic_topics = generate_dynamic_topics(vector_store, llm)
    
    if not dynamic_topics:
        logging.error("Could not generate any topics. Exiting pipeline.")
        return
        
    compiled_data = process_topics_and_extract_data(dynamic_topics, vector_store, llm)
    generate_docx_report(compiled_data, OUTPUT_DOCX_FILE)
    
    logging.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
