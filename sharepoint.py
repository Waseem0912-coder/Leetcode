#!/usr/bin/env python3
"""
Fixed RAG Pipeline with improved bullet point extraction.
Key improvements:
1. Better prompt engineering for more reliable JSON generation
2. Increased temperature for more diverse bullet points
3. Improved context handling and chunking
4. Better fallback mechanisms for parsing
5. More robust error handling
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
    """Custom embedding class for Qwen3-Embedding-0.6B model."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = None):
        """Initialize the custom Qwen embeddings."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
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
        batch_size = 8
        
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
    """Main RAG pipeline for processing PDFs and generating reports."""
    
    def __init__(self, source_dir: str = "./source_pdfs", persist_dir: str = "./chroma_db"):
        """Initialize the RAG pipeline."""
        self.source_dir = source_dir
        self.persist_dir = persist_dir
        self.vector_store = None
        self.llm = None
        self.embeddings = None
    
    def setup_llm(self):
        """Set up the Qwen3-1.7B language model with 4-bit quantization."""
        logger.info("Setting up Qwen3-1.7B with 4-bit quantization...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model_name = "Qwen/Qwen3-1.7B"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # CRITICAL FIX: Increased temperature for better generation
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,  # Increased token limit
            temperature=0.3,  # Increased from 0.1 for more diverse output
            do_sample=True,  # Changed to True for better variety
            top_p=0.9,  # Added nucleus sampling
            pad_token_id=tokenizer.pad_token_id
        )
        
        self.llm = HuggingFacePipeline(pipeline=text_pipeline)
        logger.info("LLM setup complete")
    
    def setup_embeddings(self):
        """Initialize the custom Qwen embedding model."""
        logger.info("Setting up custom Qwen embeddings...")
        self.embeddings = CustomQwenEmbeddings()
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
        
        # IMPROVED: Better chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Reduced from 1000 for more focused chunks
            chunk_overlap=150,  # Reduced overlap
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
        """Dynamically discover main topics from the document collection."""
        logger.info("Discovering topics from documents...")
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": sample_size}
        )
        
        sample_query = "main topics subjects themes content areas discussed"
        sample_docs = retriever.invoke(sample_query)
        
        context = "\n\n".join([doc.page_content for doc in sample_docs[:20]])  # Limit context
        
        # IMPROVED: Better topic discovery prompt
        topic_prompt = PromptTemplate(
            input_variables=["context"],
            template="""Based on the following text, identify 5-7 main topics or themes.

Text:
{context}

Output format: Return ONLY a JSON array of topic strings, nothing else.
Example: ["Topic A", "Topic B", "Topic C"]

JSON array:"""
        )
        
        topic_chain = topic_prompt | self.llm | StrOutputParser()
        result = topic_chain.invoke({"context": context[:6000]})
        
        try:
            result = result.strip()
            json_start = result.find('[')
            json_end = result.rfind(']') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = result[json_start:json_end]
                topics = json.loads(json_str)
                
                if isinstance(topics, list):
                    topics = [str(topic).strip() for topic in topics if str(topic).strip()]
                    logger.info(f"Discovered {len(topics)} topics: {topics}")
                    return topics
                else:
                    raise ValueError("Parsed JSON is not a list")
            else:
                raise ValueError("No JSON array found in response")
        
        except Exception as e:
            logger.warning(f"Failed to parse topics: {e}")
            logger.debug(f"Response was: {result[:200]}...")
            
            matches = re.findall(r'"([^"]*)"', result)
            if matches and len(matches) >= 3:
                logger.info(f"Extracted {len(matches)} topics using regex fallback")
                return matches[:7]
            
            default_topics = [
                "Project Status",
                "Technical Updates",
                "Budget and Resources",
                "Timeline and Milestones",
                "Risks and Issues"
            ]
            logger.info(f"Using default topics: {default_topics}")
            return default_topics
    
    def extract_topic_updates(self, topic: str) -> List[str]:
        """
        FIXED: Extract all updates related to a specific topic.
        Main improvements:
        - Better retrieval strategy
        - Improved prompts with clear instructions
        - Multiple extraction passes
        - Better parsing with fallbacks
        """
        logger.info(f"Extracting updates for topic: {topic}")
        
        # IMPROVED: Use multiple search strategies
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 25  # Get more documents
            }
        )
        
        # Search with topic-specific query
        search_queries = [
            topic,
            f"{topic} information details facts",
            f"what about {topic}",
        ]
        
        all_docs = []
        seen_content = set()
        
        # Retrieve documents using multiple queries to get diverse results
        for query in search_queries:
            docs = retriever.invoke(query)
            for doc in docs:
                # Deduplicate by content
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        logger.info(f"Retrieved {len(all_docs)} unique documents for topic: {topic}")
        
        if not all_docs:
            logger.warning(f"No documents found for topic: {topic}")
            return []
        
        all_updates = []
        
        # IMPROVED: Process documents in smaller batches with better prompts
        batch_size = 3  # Smaller batches for better focus
        
        for i in range(0, min(len(all_docs), 15), batch_size):  # Limit to first 15 docs
            batch_docs = all_docs[i:i + batch_size]
            context = "\n\n---\n\n".join([doc.page_content for doc in batch_docs])
            
            # CRITICAL FIX: Much clearer extraction prompt
            extraction_prompt = PromptTemplate(
                input_variables=["topic", "context"],
                template="""Your task: Extract specific facts and information about "{topic}" from the text below.

Text:
{context}

Instructions:
1. Read the text carefully
2. Find all sentences or facts related to "{topic}"
3. Extract each fact as a separate bullet point
4. Return ONLY a JSON array of strings
5. Each string should be one complete fact or update
6. If no relevant information found, return []

Output format example:
["First fact about the topic", "Second fact about the topic", "Third fact about the topic"]

JSON array:"""
            )
            
            extraction_chain = extraction_prompt | self.llm | StrOutputParser()
            result = extraction_chain.invoke({"topic": topic, "context": context[:5000]})
            
            # IMPROVED: Better parsing with multiple fallback strategies
            parsed_updates = self._parse_json_response(result, f"extraction batch {i//batch_size}")
            
            if parsed_updates:
                all_updates.extend(parsed_updates)
                logger.info(f"Batch {i//batch_size}: extracted {len(parsed_updates)} updates")
            else:
                # FALLBACK: Try to extract sentences directly from context
                fallback_updates = self._extract_sentences_about_topic(context, topic)
                if fallback_updates:
                    all_updates.extend(fallback_updates[:5])  # Limit fallback
                    logger.info(f"Batch {i//batch_size}: used fallback extraction, got {len(fallback_updates[:5])} updates")
        
        logger.info(f"Extracted {len(all_updates)} total updates for {topic}")
        return all_updates
    
    def _parse_json_response(self, response: str, context: str = "") -> List[str]:
        """
        Robust JSON parsing with multiple fallback strategies.
        """
        try:
            response = response.strip()
            
            # Strategy 1: Find JSON array boundaries
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                
                # Try to parse
                parsed = json.loads(json_str)
                
                if isinstance(parsed, list):
                    # Clean and validate items
                    cleaned = []
                    for item in parsed:
                        if isinstance(item, str):
                            item = item.strip()
                            if item and len(item) > 10:  # Minimum length filter
                                cleaned.append(item)
                        elif isinstance(item, dict):
                            # Sometimes LLM returns objects, try to extract values
                            for value in item.values():
                                if isinstance(value, str) and len(value) > 10:
                                    cleaned.append(value.strip())
                    
                    if cleaned:
                        logger.debug(f"{context}: Successfully parsed {len(cleaned)} items")
                        return cleaned
            
            # Strategy 2: Regex to find quoted strings
            matches = re.findall(r'"([^"]{10,})"', response)
            if matches:
                logger.debug(f"{context}: Extracted {len(matches)} items using regex")
                return [m.strip() for m in matches]
            
            # Strategy 3: Look for bullet points or numbered lists
            lines = response.split('\n')
            bullet_items = []
            for line in lines:
                line = line.strip()
                # Match various bullet formats
                match = re.match(r'^[\-\*\•\d+\.\)]\s*(.+)$', line)
                if match:
                    item = match.group(1).strip()
                    if len(item) > 10:
                        bullet_items.append(item)
            
            if bullet_items:
                logger.debug(f"{context}: Extracted {len(bullet_items)} items from bullet points")
                return bullet_items
            
            logger.warning(f"{context}: All parsing strategies failed")
            logger.debug(f"Response was: {response[:300]}...")
            return []
        
        except json.JSONDecodeError as e:
            logger.warning(f"{context}: JSON decode error: {e}")
            return []
        except Exception as e:
            logger.error(f"{context}: Unexpected error: {e}")
            return []
    
    def _extract_sentences_about_topic(self, text: str, topic: str) -> List[str]:
        """
        Fallback method: Extract sentences that mention the topic.
        """
        sentences = re.split(r'[.!?]+', text)
        relevant = []
        
        topic_words = set(topic.lower().split())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            sentence_words = set(sentence.lower().split())
            
            # Check if topic words appear in sentence
            if topic_words & sentence_words:
                relevant.append(sentence)
        
        return relevant[:10]  # Limit to 10 sentences
    
    def deduplicate_updates(self, updates: List[str]) -> List[str]:
        """Remove duplicate and semantically similar updates."""
        if not updates:
            return []
        
        # Simple exact deduplication first
        updates = list(dict.fromkeys(updates))
        
        if len(updates) <= 10:
            return updates
        
        logger.info(f"Deduplicating {len(updates)} updates...")
        
        # IMPROVED: Better deduplication approach
        deduplicated = []
        batch_size = 10
        
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            
            dedup_prompt = PromptTemplate(
                input_variables=["updates"],
                template="""Task: Remove duplicate or highly similar items from this list. Keep the most informative version.

List:
{updates}

Return ONLY a JSON array with unique items, nothing else.
Example: ["unique item 1", "unique item 2"]

JSON array:"""
            )
            
            dedup_chain = dedup_prompt | self.llm | StrOutputParser()
            updates_str = "\n".join([f"{j+1}. {u}" for j, u in enumerate(batch)])
            result = dedup_chain.invoke({"updates": updates_str})
            
            parsed = self._parse_json_response(result, f"dedup batch {i//batch_size}")
            
            if parsed:
                deduplicated.extend(parsed)
                logger.info(f"Dedup batch {i//batch_size}: {len(batch)} -> {len(parsed)} updates")
            else:
                # Fallback: keep original batch
                deduplicated.extend(batch)
                logger.warning(f"Dedup batch {i//batch_size}: using original batch")
        
        # Final exact deduplication
        final_unique = list(dict.fromkeys(deduplicated))
        logger.info(f"Reduced from {len(updates)} to {len(final_unique)} unique updates")
        
        return final_unique
    
    def generate_report(self, compiled_data: Dict[str, List[str]], output_file: str = "Consolidated_Report.docx"):
        """Generate a DOCX report from the compiled data."""
        logger.info(f"Generating report: {output_file}")
        
        doc = Document()
        
        # Add title
        title = doc.add_heading("Consolidated Report - All Topics and Updates", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add executive summary
        doc.add_heading("Executive Summary", 1)
        total_updates = sum(len(updates) for updates in compiled_data.values())
        summary = f"This report consolidates information from multiple PDF documents across {len(compiled_data)} main topics, extracting {total_updates} unique data points."
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
        
        # Add metadata
        doc.add_page_break()
        doc.add_heading("Report Metadata", 2)
        doc.add_paragraph(f"Generated using RAG pipeline with Qwen3 models")
        doc.add_paragraph(f"Total topics processed: {len(compiled_data)}")
        doc.add_paragraph(f"Total unique updates: {total_updates}")
        
        # Save document
        doc.save(output_file)
        logger.info(f"Report saved to {output_file}")
    
    def run_pipeline(self):
        """Execute the complete RAG pipeline."""
        logger.info("Starting RAG pipeline execution...")
        
        # Step 1: Setup models
        self.setup_embeddings()
        self.setup_llm()
        
        # Step 2: Ingest and index documents
        if not self.ingest_and_index_documents():
            logger.error("No documents to process. Exiting.")
            return
        
        # Step 3: Discover topics dynamically
        topics = self.discover_topics()
        
        # Step 4: Extract updates for each topic
        compiled_data = {}
        for topic in tqdm(topics, desc="Processing topics"):
            # Extract all updates for this topic
            topic_updates = self.extract_topic_updates(topic)
            
            # Deduplicate updates
            if topic_updates:
                unique_updates = self.deduplicate_updates(topic_updates)
                compiled_data[topic] = unique_updates
            else:
                compiled_data[topic] = []
            
            logger.info(f"Topic '{topic}': {len(compiled_data[topic])} unique updates")
        
        # Step 5: Generate the report
        self.generate_report(compiled_data)
        
        logger.info("Pipeline execution complete!")
        
        # Print summary
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Topics discovered: {len(topics)}")
        for topic, updates in compiled_data.items():
            print(f"  - {topic}: {len(updates)} updates")
        print(f"\nReport generated: Consolidated_Report.docx")
        print("="*50)


def main():
    """Main entry point for the script."""
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
    
    pipeline = RAGPipeline(source_dir=source_dir)
    
    try:
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
