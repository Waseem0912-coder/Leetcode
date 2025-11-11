Here's a complete Python script that implements a RAG-based approach to analyze PDFs and generate organized reports:

```python
"""
RAG-based PDF Analysis System
This script processes PDFs in a target folder, creates embeddings, and generates
organized reports of unique topics and events using separate models for embedding and inference.
"""

import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Dict, Tuple
import PyPDF2
import numpy as np
from pathlib import Path
import json

# ============================================================================
# EMBEDDING MODEL SETUP (Qwen3-Embedding-0.6B)
# ============================================================================

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last token from the hidden states."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format instruction for embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'


class EmbeddingModel:
    """Handles text embedding generation using Qwen3-Embedding-0.6B."""
    
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B'):
        print(f"Loading embedding model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = 8192
        self.device = self.model.device
        print("Embedding model loaded successfully.")
    
    def create_embeddings(self, texts: List[str], task_description: str = None) -> Tensor:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            task_description: Optional task description for queries
        
        Returns:
            Normalized embeddings tensor
        """
        # Add instruction prefix if task description provided
        if task_description:
            input_texts = [get_detailed_instruct(task_description, text) for text in texts]
        else:
            input_texts = texts
        
        # Tokenize
        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = batch_dict.to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


# ============================================================================
# INFERENCE MODEL SETUP (Qwen3-1.7B)
# ============================================================================

class InferenceModel:
    """Handles text generation using Qwen3-1.7B."""
    
    def __init__(self, model_name="Qwen/Qwen3-1.7B"):
        print(f"Loading inference model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Inference model loaded successfully.")
    
    def generate_response(self, prompt: str, max_tokens: int = 4096) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
        
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
        except ValueError:
            index = 0
        
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content


# ============================================================================
# PDF PROCESSING
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


# ============================================================================
# RAG SYSTEM
# ============================================================================

class RAGSystem:
    """RAG system for PDF analysis and report generation."""
    
    def __init__(self, pdf_folder: str = "pdf"):
        self.pdf_folder = pdf_folder
        self.embedding_model = EmbeddingModel()
        self.inference_model = InferenceModel()
        self.documents = []
        self.embeddings = None
        self.pdf_metadata = []
    
    def load_pdfs(self):
        """Load and process all PDFs in the target folder."""
        pdf_path = Path(self.pdf_folder)
        
        if not pdf_path.exists():
            print(f"Creating folder: {self.pdf_folder}")
            pdf_path.mkdir(parents=True, exist_ok=True)
            print(f"Please add PDF files to the '{self.pdf_folder}' folder and run again.")
            return
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in '{self.pdf_folder}' folder.")
            return
        
        print(f"\nFound {len(pdf_files)} PDF file(s). Processing...")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            text = extract_text_from_pdf(str(pdf_file))
            
            if text:
                # Chunk the text
                chunks = chunk_text(text, chunk_size=1500, overlap=300)
                
                for i, chunk in enumerate(chunks):
                    self.documents.append(chunk)
                    self.pdf_metadata.append({
                        'filename': pdf_file.name,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    })
        
        print(f"Loaded {len(self.documents)} text chunks from {len(pdf_files)} PDF(s).")
    
    def create_embeddings(self):
        """Create embeddings for all document chunks."""
        if not self.documents:
            print("No documents to embed. Load PDFs first.")
            return
        
        print("\nCreating embeddings for all document chunks...")
        self.embeddings = self.embedding_model.create_embeddings(self.documents)
        print(f"Created embeddings with shape: {self.embeddings.shape}")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of (chunk_text, similarity_score, metadata) tuples
        """
        if self.embeddings is None:
            print("No embeddings available. Create embeddings first.")
            return []
        
        # Create query embedding
        task = "Given a document analysis task, retrieve relevant sections that contain the information"
        query_embedding = self.embedding_model.create_embeddings([query], task_description=task)
        
        # Calculate similarities
        similarities = (query_embedding @ self.embeddings.T).squeeze(0)
        
        # Get top-k indices
        top_k_indices = torch.topk(similarities, min(top_k, len(similarities))).indices.tolist()
        
        # Return results
        results = []
        for idx in top_k_indices:
            results.append((
                self.documents[idx],
                similarities[idx].item(),
                self.pdf_metadata[idx]
            ))
        
        return results
    
    def generate_comprehensive_report(self, top_k_per_topic: int = 10):
        """
        Generate a comprehensive report analyzing all PDFs for unique topics and events.
        
        Args:
            top_k_per_topic: Number of chunks to retrieve per analysis
        """
        if not self.documents:
            print("No documents loaded. Please add PDFs and try again.")
            return
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # First pass: Identify main topics across all documents
        print("\nStep 1: Identifying main topics and sections...")
        
        topic_query = "Identify the main topics, sections, themes, and subject areas discussed"
        relevant_chunks = self.retrieve_relevant_chunks(topic_query, top_k=top_k_per_topic)
        
        context = "\n\n".join([f"[From {meta['filename']}, chunk {meta['chunk_id']+1}/{meta['total_chunks']}]\n{chunk}" 
                               for chunk, _, meta in relevant_chunks])
        
        topic_prompt = f"""Based on the following excerpts from multiple PDF documents, identify and list all the main topics, sections, or themes discussed:

{context}

Please provide a comprehensive list of all major topics and themes you can identify."""
        
        topics_response = self.inference_model.generate_response(topic_prompt, max_tokens=2048)
        print("\nIdentified Topics:")
        print(topics_response)
        
        # Second pass: For each identified topic, gather unique information
        print("\n" + "-"*80)
        print("Step 2: Analyzing each topic for unique information and events...")
        print("-"*80)
        
        # Retrieve more chunks for comprehensive analysis
        analysis_chunks = self.retrieve_relevant_chunks(
            "all unique information, events, findings, and details", 
            top_k=min(20, len(self.documents))
        )
        
        full_context = "\n\n".join([
            f"[Document: {meta['filename']}, Section {meta['chunk_id']+1}/{meta['total_chunks']}]\n{chunk}" 
            for chunk, _, meta in analysis_chunks
        ])
        
        final_prompt = f"""You are analyzing multiple PDF documents. Based on the content below, create a comprehensive report that:

1. Organizes information by topic/section
2. Lists all unique events, findings, or information under each topic
3. Avoids repetition - only mention each unique piece of information once
4. Uses clear headings and structure
5. Cites which document each piece of information comes from

Previously identified topics:
{topics_response}

Document Content:
{full_context}

Please create a well-organized report in the following format:

# COMPREHENSIVE ANALYSIS REPORT

## Topic/Section 1: [Name]
- Unique finding/event 1 (Source: filename.pdf)
- Unique finding/event 2 (Source: filename.pdf)
...

## Topic/Section 2: [Name]
- Unique finding/event 1 (Source: filename.pdf)
...

Continue for all identified topics. Be thorough and include all unique information."""
        
        print("\nGenerating final report...\n")
        final_report = self.inference_model.generate_response(final_prompt, max_tokens=8192)
        
        print("\n" + "="*80)
        print("FINAL REPORT")
        print("="*80 + "\n")
        print(final_report)
        
        # Save report to file
        output_file = "rag_analysis_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAG-BASED PDF ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analyzed {len(set([m['filename'] for m in self.pdf_metadata]))} PDF file(s)\n")
            f.write(f"Total chunks processed: {len(self.documents)}\n\n")
            f.write("-"*80 + "\n\n")
            f.write(final_report)
        
        print(f"\n\nReport saved to: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("RAG-BASED PDF ANALYSIS SYSTEM")
    print("="*80)
    
    # Initialize RAG system
    rag_system = RAGSystem(pdf_folder="pdf")
    
    # Load PDFs
    rag_system.load_pdfs()
    
    if not rag_system.documents:
        print("\nNo documents to process. Exiting.")
        return
    
    # Create embeddings
    rag_system.create_embeddings()
    
    # Generate comprehensive report
    rag_system.generate_comprehensive_report(top_k_per_topic=15)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
```

**Installation Requirements:**

Create a `requirements.txt` file:

```txt
torch>=2.0.0
transformers>=4.51.0
PyPDF2>=3.0.0
numpy>=1.24.0
```

Install with:
```bash
pip install -r requirements.txt
```

**Key Features:**

1. **Separate Models**: Uses Qwen3-Embedding-0.6B for embeddings and Qwen3-1.7B for inference
2. **RAG Architecture**: Retrieves relevant document chunks before generating responses
3. **PDF Processing**: Automatically extracts and chunks text from all PDFs in the "pdf" folder
4. **Organized Reports**: Generates structured reports with topics and unique findings
5. **Source Attribution**: Tracks which PDF each piece of information comes from
6. **Overlap Chunking**: Uses overlapping chunks to preserve context
7. **Similarity Search**: Finds most relevant document sections using cosine similarity
8. **Report Export**: Saves the final report to a text file

**Usage:**

1. Place your PDF files in a folder named `pdf` in the same directory as the script
2. Run: `python script_name.py`
3. The script will generate a comprehensive report in `rag_analysis_report.txt`

The script handles all the embedding creation, similarity search, and report generation automatically!
