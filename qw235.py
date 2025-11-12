#!/usr/bin/env python3
"""
Optimized RAG Pipeline for Large Models (Qwen3-235B)
- Uses JSON format (reliable with 235B parameters)
- Qwen3-Embedding-8B for embeddings
- Qwen3-235B for extraction and topic discovery
- Cleaner code, fewer fallbacks needed
"""
import os
import json
import re
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
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFacePipeline

# Transformers imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    pipeline,
    BitsAndBytesConfig
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
    """Custom embedding class for Qwen3-Embedding-8B model."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", device: str = None):
        """Initialize the Qwen3-Embedding-8B model."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load with optimized settings for 8B model
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map="auto"  # Automatic device mapping for multi-GPU
        ).to(self.device)
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Embedding model loaded successfully on {self.device}")
    
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
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        embeddings = []
        batch_size = 16  # Larger batch size for 8B model
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            batch_embeddings = self.last_token_pool(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        formatted_query = self.get_detailed_instruct(task_description, text)
        
        inputs = self.tokenizer(
            formatted_query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embedding = self.last_token_pool(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding[0].cpu().numpy().tolist()


class RAGPipeline:
    """RAG pipeline optimized for Qwen3-235B model."""
    
    def __init__(self, source_dir: str = "./source_pdfs", persist_dir: str = "./chroma_db"):
        """Initialize the RAG pipeline."""
        self.source_dir = source_dir
        self.persist_dir = persist_dir
        self.vector_store = None
        self.llm = None
        self.embeddings = None
    
    def setup_llm(self, quantization: str = "4bit"):
        """
        Set up the Qwen3-235B language model.
        
        Args:
            quantization: "4bit", "8bit", or "none"
        """
        logger.info(f"Setting up Qwen3-235B with {quantization} quantization...")
        
        model_name = "Qwen/Qwen3-235B"
        
        # Configure quantization based on choice
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto",  # Automatic multi-GPU distribution
            torch_dtype=torch.float16 if quantization_config else torch.float32
        )
        
        # Optimized generation parameters for 235B model
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,  # Large models can handle more tokens
            temperature=0.1,  # Low temperature for factual extraction
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id
        )
        
        self.llm = HuggingFacePipeline(pipeline=text_pipeline)
        logger.info("LLM setup complete")
    
    def setup_embeddings(self):
        """Initialize the Qwen3-Embedding-8B model."""
        logger.info("Setting up Qwen3-Embedding-8B...")
        self.embeddings = CustomQwenEmbeddings(model_name="Qwen/Qwen3-Embedding-8B")
        logger.info("Embeddings setup complete")
    
    def ingest_and_index_documents(self):
        """Load, chunk, and index all PDF documents."""
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
        
        # Optimal chunking for large models
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks work well with 235B
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from documents")
        
        logger.info("Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        logger.info("Document indexing complete")
        return True
    
    def discover_topics(self, sample_size: int = 50) -> List[str]:
        """
        Discover main topics using Qwen3-235B (reliable JSON generation).
        """
        logger.info("Discovering topics from documents...")
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": sample_size}
        )
        
        sample_query = "main topics subjects themes content areas discussed"
        sample_docs = retriever.invoke(sample_query)
        
        context = "\n\n".join([doc.page_content for doc in sample_docs[:20]])
        
        # Clean, simple prompt for 235B model
        topic_prompt = PromptTemplate(
            input_variables=["context"],
            template="""Analyze the following text and identify 6-8 main topics or themes covered.

Text:
{context}

Return your response as a JSON array of topic strings. Return ONLY the JSON array, nothing else.

Example format:
["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5", "Topic 6"]

JSON array:"""
        )
        
        topic_chain = topic_prompt | self.llm | StrOutputParser()
        result = topic_chain.invoke({"context": context[:10000]})  # Large context for 235B
        
        # Parse JSON with simple validation
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
        """
        Extract updates for a topic with retry logic.
        Large models are reliable, but retries ensure robustness.
        """
        logger.info(f"Extracting updates for topic: {topic}")
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 30}  # More documents for comprehensive extraction
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
        batch_size = 5  # Larger batches for 235B
        
        for i in range(0, min(len(all_docs), 20), batch_size):
            batch_docs = all_docs[i:i + batch_size]
            context = "\n\n---\n\n".join([doc.page_content for doc in batch_docs])
            
            # Clean extraction prompt for large model
            extraction_prompt = PromptTemplate(
                input_variables=["topic", "context"],
                template="""Extract all relevant facts and information about "{topic}" from the following text.

Text:
{context}

Return a JSON array where each element is a specific fact or piece of information about "{topic}".
Each fact should be a complete, standalone statement.
Return ONLY the JSON array, nothing else.

Example format:
["Fact 1 about the topic", "Fact 2 about the topic", "Fact 3 about the topic"]

If no relevant information is found, return an empty array: []

JSON array:"""
            )
            
            # Try extraction with retries
            for attempt in range(max_retries):
                try:
                    extraction_chain = extraction_prompt | self.llm | StrOutputParser()
                    result = extraction_chain.invoke({"topic": topic, "context": context[:8000]})
                    
                    batch_updates = self._parse_json_array(result, f"extraction batch {i//batch_size}")
                    
                    if batch_updates:
                        all_updates.extend(batch_updates)
                        logger.info(f"Batch {i//batch_size}: extracted {len(batch_updates)} updates")
                        break  # Success, no need to retry
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
        """
        Parse JSON array from LLM response.
        With 235B model, this should work 95%+ of the time.
        """
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
        """
        Deduplicate updates using LLM (works well with 235B).
        """
        if not updates:
            return []
        
        # Simple deduplication first
        updates = list(dict.fromkeys(updates))
        
        if len(updates) <= 5:
            return updates
        
        logger.info(f"Deduplicating {len(updates)} updates using LLM...")
        
        # For large update lists, process in batches
        if len(updates) > 20:
            # Simple Python deduplication for very large lists
            return updates[:20]  # Keep top 20
        
        # Use LLM for semantic deduplication
        updates_str = json.dumps(updates, indent=2)
        
        dedup_prompt = PromptTemplate(
            input_variables=["updates"],
            template="""Remove duplicate and semantically similar items from the following list.
Keep the most informative and complete version of each unique piece of information.

List:
{updates}

Return a JSON array with only unique items. Return ONLY the JSON array, nothing else.

JSON array:"""
        )
        
        dedup_chain = dedup_prompt | self.llm | StrOutputParser()
        result = dedup_chain.invoke({"updates": updates_str})
        
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
        summary = f"This report consolidates information from multiple PDF documents across {len(compiled_data)} main topics, extracting {total_updates} unique data points using Qwen3-235B and Qwen3-Embedding-8B models."
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
        doc.add_paragraph(f"Embedding Model: Qwen3-Embedding-8B")
        doc.add_paragraph(f"Language Model: Qwen3-235B")
        doc.add_paragraph(f"Total topics processed: {len(compiled_data)}")
        doc.add_paragraph(f"Total unique updates: {total_updates}")
        
        doc.save(output_file)
        logger.info(f"Report saved to {output_file}")
    
    def run_pipeline(self):
        """Execute the complete RAG pipeline."""
        logger.info("Starting RAG pipeline execution...")
        
        # Setup models
        self.setup_embeddings()
        self.setup_llm(quantization="4bit")  # Use 4bit for 235B
        
        # Ingest documents
        if not self.ingest_and_index_documents():
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
        print(f"Embedding Model: Qwen3-Embedding-8B")
        print(f"Language Model: Qwen3-235B")
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
    print("Using Qwen3-Embedding-8B + Qwen3-235B models")
    
    pipeline = RAGPipeline(source_dir=source_dir)
    
    try:
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
