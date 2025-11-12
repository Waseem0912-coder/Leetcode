#!/usr/bin/env python3
"""
Fixed RAG Pipeline for Qwen3-Next-80B-A3B-Instruct
- Properly handles Qwen3-Embedding-8B with 4096 dimensions
- Recreates vector store if dimension mismatch detected
- Multi-GPU support with automatic device mapping
"""
import os
import json
import re
import shutil
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

# Transformers imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
)

# Document generation
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomQwenEmbeddings(Embeddings):
    """Custom embedding class for Qwen3-Embedding-8B model (4096 dimensions)."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", device: str = None, use_flash_attention: bool = False):
        """Initialize the Qwen3-Embedding-8B model.
        
        Args:
            model_name: Model name or path
            device: Device to use (auto-detected if None)
            use_flash_attention: Whether to use flash_attention_2 (requires installation)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        
        # Check available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Found {num_gpus} GPU(s) available for embeddings")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # CRITICAL: Use left padding for Qwen3-Embedding-8B
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left',  # Left padding as per official implementation
            trust_remote_code=True
        )
        
        # Load model with optional flash attention
        model_kwargs = {
            'trust_remote_code': True,
            'device_map': 'auto',  # Automatic multi-GPU distribution
        }
        
        if use_flash_attention and self.device == 'cuda':
            logger.info("Enabling flash_attention_2 for better performance")
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            model_kwargs['torch_dtype'] = torch.float16
        elif self.device == 'cuda':
            model_kwargs['torch_dtype'] = torch.float16
        else:
            model_kwargs['torch_dtype'] = torch.float32
        
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        
        # Maximum sequence length for Qwen3-Embedding-8B
        self.max_length = 8192
        
        # Get embedding dimension
        with torch.no_grad():
            test_input = self.tokenizer(["test"], padding=True, truncation=True, 
                                       max_length=self.max_length, return_tensors="pt")
            test_input = {k: v.to(self.model.device) for k, v in test_input.items()}
            test_output = self.model(**test_input)
            test_emb = self.last_token_pool(test_output.last_hidden_state, test_input['attention_mask'])
            self.embedding_dim = test_emb.shape[1]
        
        logger.info(f"Embedding model loaded successfully")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Max sequence length: {self.max_length}")
    
    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool the last token from the sequence."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        """Format the query with task-specific instructions."""
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (no instruction prefix for documents)."""
        if not texts:
            return []
        
        embeddings = []
        batch_size = 16  # Larger batch size for 8B model
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize with proper settings
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            
            # Apply last token pooling
            batch_embeddings = self.last_token_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )
            
            # Normalize embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text with instruction prefix."""
        # Add instruction for queries (not for documents)
        task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        formatted_query = self.get_detailed_instruct(task_description, text)
        
        # Tokenize
        batch_dict = self.tokenizer(
            [formatted_query],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        
        # Get embedding
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        
        # Apply last token pooling
        embedding = self.last_token_pool(
            outputs.last_hidden_state,
            batch_dict['attention_mask']
        )
        
        # Normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding[0].cpu().numpy().tolist()


class Qwen3NextLLM:
    """Wrapper for Qwen3-Next-80B model with proper chat template support."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"):
        """Initialize Qwen3-Next-80B model."""
        logger.info(f"Loading language model: {model_name}")
        
        # Check available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Found {num_gpus} GPU(s) available for LLM")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with automatic multi-GPU distribution
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",  # Automatic dtype selection
            device_map="auto",   # Automatic multi-GPU distribution
            trust_remote_code=True
        )
        
        self.model.eval()
        
        logger.info(f"Language model loaded successfully")
        logger.info(f"Model device map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'Single device'}")
    
    def generate(self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.1) -> str:
        """
        Generate text using Qwen3-Next with proper chat template.
        
        Args:
            prompt: The user prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text response
        """
        # Format using chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt")
        
        # Move to model device (first GPU if multi-GPU)
        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        # Store input length for later (before passing dict to generate)
        input_length = model_inputs['input_ids'].shape[1]
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part (skip the input prompt)
        output_ids = generated_ids[0][input_length:].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return content.strip()
    
    def invoke(self, input_dict: Dict[str, str]) -> str:
        """LangChain-compatible invoke method."""
        # Extract the prompt from input
        if isinstance(input_dict, dict):
            # Reconstruct prompt from template variables
            prompt = str(input_dict)
        else:
            prompt = str(input_dict)
        
        return self.generate(prompt)


class RAGPipeline:
    """RAG pipeline optimized for Qwen3-Next-80B with multi-GPU support."""
    
    def __init__(self, source_dir: str = "./source_pdfs", persist_dir: str = "./chroma_db"):
        """Initialize the RAG pipeline."""
        self.source_dir = source_dir
        self.persist_dir = persist_dir
        self.vector_store = None
        self.llm = None
        self.embeddings = None
    
    def setup_llm(self):
        """Set up the Qwen3-Next-80B language model."""
        logger.info("Setting up Qwen3-Next-80B...")
        self.llm = Qwen3NextLLM(model_name="Qwen/Qwen3-Next-80B-A3B-Instruct")
        logger.info("LLM setup complete")
    
    def setup_embeddings(self, use_flash_attention: bool = False):
        """Initialize the Qwen3-Embedding-8B model.
        
        Args:
            use_flash_attention: Enable flash_attention_2 for better performance (requires flash-attn package)
        """
        logger.info("Setting up Qwen3-Embedding-8B...")
        self.embeddings = CustomQwenEmbeddings(
            model_name="Qwen/Qwen3-Embedding-8B",
            use_flash_attention=use_flash_attention
        )
        logger.info("Embeddings setup complete")
    
    def _check_vector_store_compatibility(self) -> bool:
        """Check if existing vector store is compatible with current embeddings."""
        if not os.path.exists(self.persist_dir):
            return True  # No existing store
        
        try:
            # Try to load existing vector store
            test_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            
            # Try a simple query to check compatibility
            test_store.similarity_search("test", k=1)
            logger.info("Existing vector store is compatible")
            return True
        
        except Exception as e:
            logger.warning(f"Vector store compatibility check failed: {e}")
            logger.warning("This usually means dimension mismatch between old and new embeddings")
            return False
    
    def ingest_and_index_documents(self, force_recreate: bool = False):
        """Load, chunk, and index all PDF documents.
        
        Args:
            force_recreate: Force recreation of vector store even if it exists
        """
        logger.info(f"Loading PDFs from {self.source_dir}...")
        
        if not os.path.exists(self.source_dir):
            os.makedirs(self.source_dir)
            logger.warning(f"Created empty source directory: {self.source_dir}")
            logger.warning("Please add PDF files to this directory and run again.")
            return False
        
        loader = PyPDFDirectoryLoader(self.source_dir)
        documents = loader.load()
        
        if not documents:
            logger.warning(f"No PDF documents found in {self.source_dir}")
            return False
        
        logger.info(f"Loaded {len(documents)} pages from PDFs")
        
        # Check if we need to recreate vector store
        needs_recreation = force_recreate or not self._check_vector_store_compatibility()
        
        if needs_recreation:
            if os.path.exists(self.persist_dir):
                logger.warning(f"Removing incompatible vector store at {self.persist_dir}")
                logger.warning("This is normal when switching between different embedding models")
                shutil.rmtree(self.persist_dir)
            
            logger.info("Creating new vector store...")
            
            # Optimal chunking for large models
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from documents")
            
            logger.info(f"Creating vector store with {self.embeddings.embedding_dim}-dimensional embeddings...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            
            logger.info("Document indexing complete")
        else:
            logger.info("Loading existing compatible vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            logger.info("Vector store loaded successfully")
        
        return True
    
    def discover_topics(self, sample_size: int = 50) -> List[str]:
        """Discover main topics using Qwen3-Next-80B."""
        logger.info("Discovering topics from documents...")
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": sample_size}
        )
        
        sample_query = "main topics subjects themes content areas discussed"
        sample_docs = retriever.invoke(sample_query)
        
        context = "\n\n".join([doc.page_content for doc in sample_docs[:20]])
        
        # Create prompt for topic discovery
        prompt = f"""Analyze the following text and identify 6-8 main topics or themes covered.

Text:
{context[:10000]}

Return your response as a JSON array of topic strings. Return ONLY the JSON array, nothing else.

Example format:
["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5", "Topic 6"]

JSON array:"""
        
        # Generate using Qwen3-Next
        result = self.llm.generate(prompt, max_new_tokens=512, temperature=0.1)
        
        # Parse JSON
        topics = self._parse_json_array(result, "topic discovery")
        
        if not topics or len(topics) < 3:
            logger.warning("Topic discovery returned insufficient results, using fallback")
            topics = [
                "Project Overview and Status",
                "Technical Implementation",
                "Budget and Financial Planning",
                "Timeline and Milestones",
                "Team and Resources",
                "Risks and Challenges",
                "Quality and Compliance"
            ]
        
        logger.info(f"Discovered {len(topics)} topics: {topics}")
        return topics
    
    def extract_topic_updates(self, topic: str, max_retries: int = 2) -> List[str]:
        """Extract updates for a topic with retry logic."""
        logger.info(f"Extracting updates for topic: {topic}")
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 30}
        )
        
        # Enhanced search with multiple queries
        search_queries = [
            topic,
            f"{topic} details information",
            f"information about {topic}",
        ]
        
        all_docs = []
        seen_content = set()
        
        for query in search_queries:
            docs = retriever.invoke(query)
            for doc in docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        logger.info(f"Retrieved {len(all_docs)} unique documents for topic: {topic}")
        
        if not all_docs:
            return []
        
        all_updates = []
        batch_size = 5
        
        for i in range(0, min(len(all_docs), 20), batch_size):
            batch_docs = all_docs[i:i + batch_size]
            context = "\n\n---\n\n".join([doc.page_content for doc in batch_docs])
            
            # Create extraction prompt
            prompt = f"""Extract all relevant facts and information about "{topic}" from the following text.

Text:
{context[:8000]}

Return a JSON array where each element is a specific fact or piece of information about "{topic}".
Each fact should be a complete, standalone statement.
Return ONLY the JSON array, nothing else.

Example format:
["Fact 1 about the topic", "Fact 2 about the topic", "Fact 3 about the topic"]

If no relevant information is found, return an empty array: []

JSON array:"""
            
            # Try extraction with retries
            for attempt in range(max_retries):
                try:
                    result = self.llm.generate(prompt, max_new_tokens=1024, temperature=0.1)
                    batch_updates = self._parse_json_array(result, f"extraction batch {i//batch_size}")
                    
                    if batch_updates:
                        all_updates.extend(batch_updates)
                        logger.info(f"Batch {i//batch_size}: extracted {len(batch_updates)} updates")
                        break
                    elif attempt < max_retries - 1:
                        logger.warning(f"Batch {i//batch_size}: empty result, retrying...")
                        continue
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Batch {i//batch_size}: error on attempt {attempt + 1}, retrying: {e}")
                        continue
                    else:
                        logger.error(f"Batch {i//batch_size}: failed after {max_retries} attempts")
        
        logger.info(f"Extracted {len(all_updates)} total updates for {topic}")
        return all_updates
    
    def _parse_json_array(self, response: str, context: str = "") -> List[str]:
        """Parse JSON array from LLM response."""
        try:
            response = response.strip()
            
            # Remove common wrappers
            response = response.replace('```json', '').replace('```', '').strip()
            
            # Find JSON array boundaries
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                if isinstance(parsed, list):
                    # Clean and validate items
                    cleaned = []
                    for item in parsed:
                        if isinstance(item, str):
                            item = item.strip()
                            if item and len(item) > 10:
                                cleaned.append(item)
                    
                    logger.info(f"{context}: Successfully parsed {len(cleaned)} items")
                    return cleaned
                else:
                    logger.warning(f"{context}: Parsed value is not a list")
            else:
                logger.warning(f"{context}: No JSON array found in response")
            
            return []
        
        except json.JSONDecodeError as e:
            logger.error(f"{context}: JSON decode error: {e}")
            logger.debug(f"Response was: {response[:300]}...")
            return []
        except Exception as e:
            logger.error(f"{context}: Unexpected error: {e}")
            return []
    
    def deduplicate_updates(self, updates: List[str]) -> List[str]:
        """Deduplicate updates using LLM."""
        if not updates:
            return []
        
        # Simple deduplication first
        updates = list(dict.fromkeys(updates))
        
        if len(updates) <= 5:
            return updates
        
        logger.info(f"Deduplicating {len(updates)} updates using LLM...")
        
        # For large update lists, just use simple dedup
        if len(updates) > 20:
            return updates[:20]
        
        # Use LLM for semantic deduplication
        updates_str = json.dumps(updates, indent=2)
        
        prompt = f"""Remove duplicate and semantically similar items from the following list.
Keep the most informative and complete version of each unique piece of information.

List:
{updates_str}

Return a JSON array with only unique items. Return ONLY the JSON array, nothing else.

JSON array:"""
        
        result = self.llm.generate(prompt, max_new_tokens=1024, temperature=0.1)
        deduplicated = self._parse_json_array(result, "deduplication")
        
        if deduplicated and len(deduplicated) > 0:
            logger.info(f"Reduced from {len(updates)} to {len(deduplicated)} unique updates")
            return deduplicated
        else:
            logger.warning("Deduplication failed, returning original list")
            return updates
    
    def generate_report(self, compiled_data: Dict[str, List[str]], output_file: str = "Consolidated_Report.docx"):
        """Generate a DOCX report from the compiled data."""
        logger.info(f"Generating report: {output_file}")
        
        doc = Document()
        
        # Title
        title = doc.add_heading("Consolidated Report - All Topics and Updates", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Executive summary
        doc.add_heading("Executive Summary", 1)
        total_updates = sum(len(updates) for updates in compiled_data.values())
        summary = f"This report consolidates information from multiple PDF documents across {len(compiled_data)} main topics, extracting {total_updates} unique data points using Qwen3-Next-80B and Qwen3-Embedding-8B models."
        doc.add_paragraph(summary)
        doc.add_paragraph()
        
        # Add each topic and its updates
        for topic, updates in compiled_data.items():
            doc.add_heading(topic, 1)
            
            if updates:
                for update in updates:
                    p = doc.add_paragraph(style='List Bullet')
                    p.add_run(update)
            else:
                doc.add_paragraph("No updates found for this topic.", style='Body Text')
            
            doc.add_paragraph()
        
        # Metadata
        doc.add_page_break()
        doc.add_heading("Report Metadata", 2)
        doc.add_paragraph(f"Generated using RAG pipeline")
        doc.add_paragraph(f"Embedding Model: Qwen3-Embedding-8B ({self.embeddings.embedding_dim} dimensions)")
        doc.add_paragraph(f"Language Model: Qwen3-Next-80B-A3B-Instruct")
        doc.add_paragraph(f"Total topics processed: {len(compiled_data)}")
        doc.add_paragraph(f"Total unique updates: {total_updates}")
        
        doc.save(output_file)
        logger.info(f"Report saved to {output_file}")
    
    def run_pipeline(self, use_flash_attention: bool = False, force_recreate_vectorstore: bool = False):
        """Execute the complete RAG pipeline.
        
        Args:
            use_flash_attention: Enable flash_attention_2 for embeddings (requires flash-attn package)
            force_recreate_vectorstore: Force recreation of vector store even if it exists
        """
        logger.info("Starting RAG pipeline execution...")
        
        # Setup models
        self.setup_embeddings(use_flash_attention=use_flash_attention)
        self.setup_llm()
        
        # Ingest documents (will handle dimension mismatch automatically)
        if not self.ingest_and_index_documents(force_recreate=force_recreate_vectorstore):
            logger.error("No documents to process. Exiting.")
            return
        
        # Discover topics
        topics = self.discover_topics()
        
        # Extract updates for each topic
        compiled_data = {}
        for topic in tqdm(topics, desc="Processing topics"):
            topic_updates = self.extract_topic_updates(topic)
            
            if topic_updates:
                unique_updates = self.deduplicate_updates(topic_updates)
                compiled_data[topic] = unique_updates
            else:
                compiled_data[topic] = []
            
            logger.info(f"Topic '{topic}': {len(compiled_data[topic])} unique updates")
        
        # Generate report
        self.generate_report(compiled_data)
        
        logger.info("Pipeline execution complete!")
        
        # Summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Embedding Model: Qwen3-Embedding-8B ({self.embeddings.embedding_dim}D)")
        print(f"Language Model: Qwen3-Next-80B-A3B-Instruct")
        print(f"Topics discovered: {len(topics)}")
        for topic, updates in compiled_data.items():
            print(f"  - {topic}: {len(updates)} updates")
        print(f"\nReport generated: Consolidated_Report.docx")
        print("="*60)


def main():
    """Main entry point."""
    source_dir = "./source_pdfs"
    
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"Created {source_dir} directory.")
        print("Please add your PDF files to this directory and run the script again.")
        return
    
    pdf_files = list(Path(source_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        print("Please add PDF files to process and run the script again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    print("Using Qwen3-Embedding-8B + Qwen3-Next-80B-A3B-Instruct models")
    
    # Print GPU information
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\nDetected {num_gpus} GPU(s):")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        print()
    
    pipeline = RAGPipeline(source_dir=source_dir)
    
    try:
        # Run pipeline (will auto-detect and fix dimension mismatch)
        pipeline.run_pipeline(use_flash_attention=False, force_recreate_vectorstore=False)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
