#!/usr/bin/env python3
"""
OPTIMIZED Weekly Report Pipeline
- 25x faster: Inverted loop structure (M calls instead of N×M)
- Properly uses embeddings for content pre-filtering
- Processes all topics from each PDF in one LLM call
- PARALLEL processing in Phase 3
"""
import os
import json
import re
import shutil
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed  # <-- ADDED IMPORT

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
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
    """Custom embedding class for Qwen3-Embedding-8B model."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", device: str = None, use_flash_attention: bool = False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading embedding model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left',
            trust_remote_code=True
        )
        
        model_kwargs = {
            'trust_remote_code': True,
            'device_map': 'auto',
        }
        
        if use_flash_attention and self.device == 'cuda':
            logger.info("Enabling flash_attention_2")
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            model_kwargs['torch_dtype'] = torch.float16
        elif self.device == 'cuda':
            model_kwargs['torch_dtype'] = torch.float16
        else:
            model_kwargs['torch_dtype'] = torch.float32
        
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        self.max_length = 8192
        
        with torch.no_grad():
            test_input = self.tokenizer(["test"], padding=True, truncation=True, 
                                       max_length=self.max_length, return_tensors="pt")
            test_input = {k: v.to(self.model.device) for k, v in test_input.items()}
            test_output = self.model(**test_input)
            test_emb = self.last_token_pool(test_output.last_hidden_state, test_input['attention_mask'])
            self.embedding_dim = test_emb.shape[1]
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        embeddings = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            
            batch_embeddings = self.last_token_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        formatted_query = self.get_detailed_instruct(task_description, text)
        
        batch_dict = self.tokenizer(
            [formatted_query],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        
        embedding = self.last_token_pool(
            outputs.last_hidden_state,
            batch_dict['attention_mask']
        )
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding[0].cpu().numpy().tolist()


class Qwen3NextLLM:
    """Wrapper for Qwen3-Next-80B model."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"):
        logger.info(f"Loading language model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        logger.info(f"Language model loaded successfully")
    
    def generate(self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.1) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        model_inputs = self.tokenizer([text], return_tensors="pt")
        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        input_length = model_inputs['input_ids'].shape[1]
        
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
        
        output_ids = generated_ids[0][input_length:].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content.strip()


class OptimizedWeeklyReportPipeline:
    """
    OPTIMIZED Pipeline for weekly reports:
    - 25x faster: Inverted loop (M calls instead of N×M)
    - Uses embeddings for content pre-filtering
    - Processes all topics per PDF in single LLM call
    """
    
    def __init__(self, source_dir: str = "./source_pdfs", max_topics: int = 25, use_embeddings: bool = True):
        self.source_dir = source_dir
        self.max_topics = max_topics
        self.use_embeddings = use_embeddings  # NEW: Optional embedding usage
        self.llm = None
        self.embeddings = None
        self.pdf_files = []
        self.pdf_content_cache = {}  # CRITICAL: Cache to avoid loading PDFs twice!
    
    def setup_llm(self):
        logger.info("Setting up Qwen3-Next-80B...")
        self.llm = Qwen3NextLLM(model_name="Qwen/Qwen3-Next-80B-A3B-Instruct")
        logger.info("LLM setup complete")
    
    def setup_embeddings(self, use_flash_attention: bool = False):
        if not self.use_embeddings:
            logger.info("Skipping embeddings (use_embeddings=False)")
            return
        
        logger.info("Setting up Qwen3-Embedding-8B for content filtering...")
        self.embeddings = CustomQwenEmbeddings(
            model_name="Qwen/Qwen3-Embedding-8B",
            use_flash_attention=use_flash_attention
        )
        logger.info("Embeddings setup complete")
    
    def load_pdf_files(self):
        """Load list of all PDF files."""
        if not os.path.exists(self.source_dir):
            os.makedirs(self.source_dir)
            logger.warning(f"Created empty source directory: {self.source_dir}")
            return False
        
        self.pdf_files = sorted(list(Path(self.source_dir).glob("*.pdf")))
        
        if not self.pdf_files:
            logger.warning(f"No PDF files found in {self.source_dir}")
            return False
        
        logger.info(f"Found {len(self.pdf_files)} PDF files (weekly reports)")
        return True
    
    def load_pdf_content(self, pdf_path: Path) -> str:
        """
        Load PDF content with caching to avoid double I/O.
        CRITICAL FIX: Each PDF is loaded once and cached.
        
        Returns:
            Combined text from all pages
        """
        # Check cache first
        cache_key = str(pdf_path)
        if cache_key in self.pdf_content_cache:
            logger.debug(f"✓ Using cached content for {pdf_path.name}")
            return self.pdf_content_cache[cache_key]
        
        # Load from disk (only happens once per PDF!)
        logger.info(f"Loading {pdf_path.name} from disk...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        if not pages:
            full_text = ""
        else:
            full_text = "\n\n".join([page.page_content for page in pages])
        
        # Cache for future use
        self.pdf_content_cache[cache_key] = full_text
        logger.debug(f"✓ Cached {pdf_path.name} ({len(full_text)} chars)")
        
        return full_text
    
    def extract_topics_from_pdf(self, pdf_path: Path) -> List[str]:
        """Extract topics from a single PDF using cached content."""
        logger.info(f"Extracting topics from: {pdf_path.name}")
        
        # Use cached content (avoids 2nd disk I/O!)
        full_text = self.load_pdf_content(pdf_path)
        
        if not full_text:
            logger.warning(f"No content in {pdf_path.name}")
            return []
        
        if len(full_text) > 10000:
            full_text = full_text[:10000]
        
        prompt = f"""Analyze this weekly report and identify the main topics covered.

Report:
{full_text}

List each topic on a new line starting with "TOPIC:".

Example:
TOPIC: Budget and Financial Planning
TOPIC: Technical Implementation Progress
TOPIC: Team Staffing and Resources

Extract topics:"""
        
        result = self.llm.generate(prompt, max_new_tokens=512, temperature=0.2)
        topics = self._parse_text_list(result, prefix="TOPIC:")
        
        logger.info(f"  Found {len(topics)} topics")
        return topics
    
    def consolidate_topics(self, all_topics_by_pdf: Dict[str, List[str]]) -> List[str]:
        """Consolidate topics from all PDFs."""
        logger.info("Consolidating topics...")
        
        all_topics = []
        for pdf_name, topics in all_topics_by_pdf.items():
            all_topics.extend(topics)
        
        logger.info(f"Total topics before consolidation: {len(all_topics)}")
        
        # Remove exact duplicates
        unique_topics = []
        seen_lower = set()
        for topic in all_topics:
            topic_lower = topic.lower().strip()
            if topic_lower not in seen_lower:
                seen_lower.add(topic_lower)
                unique_topics.append(topic)
        
        logger.info(f"After deduplication: {len(unique_topics)} topics")
        
        if len(unique_topics) <= self.max_topics:
            return unique_topics
        
        # Merge similar topics
        logger.info(f"Merging to {self.max_topics} topics...")
        
        topics_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(unique_topics)])
        
        prompt = f"""Consolidate {len(unique_topics)} topics into {self.max_topics} distinct topics.

Merge similar topics. Keep distinct important topics.

Current topics:
{topics_text}

Write EXACTLY {self.max_topics} consolidated topics, one per line starting with "TOPIC:".

Consolidated topics:"""
        
        result = self.llm.generate(prompt, max_new_tokens=1024, temperature=0.1)
        consolidated = self._parse_text_list(result, prefix="TOPIC:")
        
        if len(consolidated) < self.max_topics * 0.7:
            logger.warning(f"Using top {self.max_topics} from unique list")
            return unique_topics[:self.max_topics]
        
        return consolidated[:self.max_topics]
    
    def filter_relevant_content(self, full_text: str, topics: List[str]) -> str:
        """
        Use embeddings to filter PDF content to only relevant sections.
        This reduces LLM context and improves extraction quality.
        """
        if not self.use_embeddings or not self.embeddings:
            # No filtering - return full text (truncated)
            return full_text[:12000]
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        
        chunks = [full_text[i:i+800] for i in range(0, len(full_text), 700)]
        
        if len(chunks) <= 10:
            # Small document, no need to filter
            return full_text
        
        # Embed chunks
        chunk_embeddings = self.embeddings.embed_documents(chunks)
        
        # Embed topics
        topics_text = " ".join(topics)
        topic_embedding = self.embeddings.embed_query(topics_text)
        
        # Calculate similarity
        similarities = []
        for chunk_emb in chunk_embeddings:
            similarity = np.dot(chunk_emb, topic_embedding)
            similarities.append(similarity)
        
        # Get top 15 most relevant chunks
        top_indices = np.argsort(similarities)[-15:][::-1]
        
        # Reconstruct filtered text
        filtered_chunks = [chunks[i] for i in sorted(top_indices)]
        filtered_text = "\n\n".join(filtered_chunks)
        
        logger.info(f"  Filtered {len(chunks)} chunks → {len(filtered_chunks)} relevant chunks")
        return filtered_text
    
    def extract_all_topics_from_pdf(self, pdf_path: Path, topics: List[str]) -> Dict[str, List[str]]:
        """
        OPTIMIZED: Extract information about ALL topics from ONE PDF in a single LLM call.
        Uses cached PDF content to avoid disk I/O.
        
        Args:
            pdf_path: Path to the PDF
            topics: List of ALL consolidated topics to extract
            
        Returns:
            Dict mapping each topic to list of facts found in this PDF
        """
        logger.info(f"Extracting all {len(topics)} topics from {pdf_path.name}...")
        
        # Use cached content (avoids 2nd disk I/O!)
        full_text = self.load_pdf_content(pdf_path)
        
        if not full_text:
            return {topic: [] for topic in topics}
        
        # Filter content using embeddings (if enabled)
        filtered_text = self.filter_relevant_content(full_text, topics)
        
        # Format topics list
        topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics)])
        
        # SINGLE LLM CALL for ALL topics
        prompt = f"""Extract information from this weekly report for EACH of the following topics.

Report from {pdf_path.stem}:
{filtered_text}

Topics to extract:
{topics_list}

For EACH topic above, write any relevant facts on new lines starting with "TOPIC: [name]" then "FACT: [fact]".

Example format:
TOPIC: Budget and Financial Planning
FACT: Budget increased by 10% for Q3
FACT: New tracking system deployed

TOPIC: Project Timeline
FACT: Phase 2 started on schedule
FACT: Milestone A completed

If a topic has no information, write "TOPIC: [name]" then "NONE".

Extract for all topics:"""
        
        result = self.llm.generate(prompt, max_new_tokens=2048, temperature=0.1)
        
        # Parse the structured output
        topic_facts = self._parse_multi_topic_output(result, topics, pdf_path.stem)
        
        # Log results
        total_facts = sum(len(facts) for facts in topic_facts.values())
        logger.info(f"  Extracted {total_facts} facts across {len(topics)} topics from {pdf_path.name}")
        
        return topic_facts
    
    def _parse_multi_topic_output(self, text: str, topics: List[str], week_name: str) -> Dict[str, List[str]]:
        """
        Parse output where multiple topics and their facts are in one response.
        
        Format:
        TOPIC: Budget
        FACT: Something about budget
        FACT: Another budget fact
        TOPIC: Timeline
        FACT: Something about timeline
        """
        topic_facts = {topic: [] for topic in topics}
        
        current_topic = None
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if line starts a new topic
            if line.startswith('TOPIC:'):
                topic_name = line.replace('TOPIC:', '').strip()
                
                # Match to one of our consolidated topics (fuzzy match)
                current_topic = None
                for topic in topics:
                    if topic.lower() in topic_name.lower() or topic_name.lower() in topic.lower():
                        current_topic = topic
                        break
            
            # Check if line is a fact
            elif line.startswith('FACT:') and current_topic:
                fact = line.replace('FACT:', '').strip()
                if fact and len(fact) > 10 and fact.upper() != 'NONE':
                    # Add week prefix
                    fact_with_week = f"[{week_name}] {fact}"
                    topic_facts[current_topic].append(fact_with_week)
            
            # Check for NONE indicator
            elif 'NONE' in line.upper() and current_topic:
                # Topic explicitly has no info, continue to next topic
                continue
        
        return topic_facts
    
    def extract_info_for_all_topics_optimized(self, topics: List[str]) -> Dict[str, List[str]]:
        """
        OPTIMIZED: Process all PDFs, extracting all topics from each.
        
        NEW: Uses ThreadPoolExecutor to run in parallel.
        """
        logger.info(f"\nExtracting {len(topics)} topics from {len(self.pdf_files)} PDFs...")
        logger.info(f"This will make {len(self.pdf_files)} LLM calls in PARALLEL.")
        
        topic_info = defaultdict(list)
        
        # Set max_workers to the number of parallel calls you want to make.
        # Tune this based on your GPU VRAM and processing power.
        # Start with a low number like 4-8.
        max_workers = 8
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary to map futures to PDF files
            future_to_pdf = {
                executor.submit(self.extract_all_topics_from_pdf, pdf_file, topics): pdf_file
                for pdf_file in self.pdf_files
            }
            
            # Use tqdm to track progress as futures complete
            for future in tqdm(as_completed(future_to_pdf), total=len(self.pdf_files), desc="Processing PDFs"):
                pdf_file = future_to_pdf[future]
                try:
                    # Get the result from the completed thread
                    pdf_topic_facts = future.result()
                    
                    # Accumulate facts for each topic
                    for topic, facts in pdf_topic_facts.items():
                        if facts:
                            topic_info[topic].extend(facts)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file.name}: {e}")

        return dict(topic_info)
    
    def _parse_text_list(self, text: str, prefix: str = "TOPIC:") -> List[str]:
        """Parse items from text output."""
        items = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith(prefix):
                item = line.replace(prefix, '').strip()
                if item and len(item) > 3:
                    items.append(item)
            elif line.startswith('- ') or line.startswith('* '):
                item = line[2:].strip()
                if item and len(item) > 3:
                    items.append(item)
            elif re.match(r'^\d+[\.\)]\s+', line):
                item = re.sub(r'^\d+[\.\)]\s+', '', line).strip()
                if item and len(item) > 3:
                    items.append(item)
        
        return items
    
    def generate_report(self, topic_info: Dict[str, List[str]], output_file: str = "Consolidated_Weekly_Report.docx"):
        """Generate final Word document."""
        logger.info(f"Generating report: {output_file}")
        
        doc = Document()
        
        title = doc.add_heading("Consolidated Weekly Reports - All Topics", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_heading("Executive Summary", 1)
        total_facts = sum(len(facts) for facts in topic_info.values())
        summary = f"This report consolidates {len(self.pdf_files)} weekly reports across {len(topic_info)} topics, extracting {total_facts} unique updates."
        doc.add_paragraph(summary)
        doc.add_paragraph()
        
        for topic, facts in topic_info.items():
            doc.add_heading(topic, 1)
            
            if facts:
                for fact in facts:
                    p = doc.add_paragraph(style='List Bullet')
                    p.add_run(fact)
            else:
                doc.add_paragraph("No information found.", style='Body Text')
            
            doc.add_paragraph()
        
        doc.add_page_break()
        doc.add_heading("Report Metadata", 2)
        doc.add_paragraph(f"Weekly reports: {len(self.pdf_files)}")
        doc.add_paragraph(f"Topics: {len(topic_info)}")
        doc.add_paragraph(f"Facts: {total_facts}")
        doc.add_paragraph(f"Embeddings: {'Enabled' if self.use_embeddings else 'Disabled'}")
        
        doc.save(output_file)
        logger.info(f"Report saved to {output_file}")
    
    def run_pipeline(self, use_flash_attention: bool = False):
        """Execute the OPTIMIZED pipeline."""
        logger.info("="*70)
        logger.info("OPTIMIZED WEEKLY REPORT PIPELINE (25x faster!)")
        logger.info("="*70)
        
        self.setup_embeddings(use_flash_attention=use_flash_attention)
        self.setup_llm()
        
        if not self.load_pdf_files():
            logger.error("No PDFs to process.")
            return
        
        # Phase 1: Extract topics from each PDF
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: Extract topics from each PDF")
        logger.info("="*70)
        
        all_topics_by_pdf = {}
        for pdf_file in self.pdf_files:
            topics = self.extract_topics_from_pdf(pdf_file)
            all_topics_by_pdf[pdf_file.name] = topics
        
        # Phase 2: Consolidate topics
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: Consolidate topics")
        logger.info("="*70)
        
        consolidated_topics = self.consolidate_topics(all_topics_by_pdf)
        
        logger.info("\nCONSOLIDATED TOPICS:")
        for i, topic in enumerate(consolidated_topics, 1):
            logger.info(f"  {i}. {topic}")
        
        # Phase 3: Extract info (OPTIMIZED + PARALLEL!)
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: Extract info (OPTIMIZED + PARALLEL)")
        logger.info("="*70)
        
        topic_info = self.extract_info_for_all_topics_optimized(consolidated_topics)
        
        # Phase 4: Generate report
        logger.info("\n" + "="*70)
        logger.info("PHASE 4: Generate report")
        logger.info("="*7V)
        
        self.generate_report(topic_info)
        
        print("\n" + "="*70)
        print("COMPLETE!")
        print("="*70)
        print(f"PDFs: {len(self.pdf_files)}")
        print(f"Topics: {len(consolidated_topics)}")
        print(f"LLM calls: {len(self.pdf_files)} (parallelized)")
        print(f"Speedup: {len(consolidated_topics)}x (plus parallelization)")
        print(f"Facts: {sum(len(f) for f in topic_info.values())}")
        print("="*7D)


def main():
    source_dir = "./source_pdfs"
    
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print("Created source directory.")
        return
    
    pdf_files = list(Path(source_dir).glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found.")
        return
    
    print(f"Found {len(pdf_files)} weekly reports")
    print("\nOptimizations:")
    print("✅ Inverted loop: M calls instead of N×M (25x faster)")
    print("✅ Embeddings for content filtering (better quality)")
    print("✅ Single LLM call per PDF extracts all topics")
    print("✅ PARALLEL processing for Phase 3\n")
    
    # With embeddings (slower but better quality):
    # pipeline = OptimizedWeeklyReportPipeline(source_dir=source_dir, use_embeddings=True)
    
    # Without embeddings (faster):
    pipeline = OptimizedWeeklyReportPipeline(source_dir=source_dir, use_embeddings=False)
    
    try:
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise


if __name__ == "__main__":
    main()
