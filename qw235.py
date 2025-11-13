#!/usr/bin/env python3
"""
OPTIMIZED Weekly Report Pipeline
- 25x faster: Inverted loop structure (M calls instead of N×M)
- Properly uses embeddings for content pre-filtering
- Processes all topics from each PDF in one LLM call
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

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

# Transformers imports
from transformers import (
    pipeline, # Using the high-level pipeline API
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


# MODIFIED: Replaced Qwen3NextLLM with GptOssLLM
class GptOssLLM:
    """Wrapper for openai/gpt-oss-120b model using transformers.pipeline."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        logger.info(f"Loading language model: {model_name}")
        
        # Initialize the pipeline as per the provided example
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        
        logger.info(f"Language model pipeline loaded successfully")
    
    def generate(self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.1) -> str:
        # The pipeline API expects a list of dictionaries for chat-like interactions
        messages = [{"role": "user", "content": prompt}]
        
        # Define generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": 0.9 if temperature > 0 else None,
            "repetition_penalty": 1.05,
        }
        
        # The pipeline automatically handles tokenization and device placement
        outputs = self.pipe(messages, **generation_kwargs)
        
        # The output format for chat inputs is a list containing the full chat history.
        # We need the content of the last message, which is the assistant's reply.
        # Structure: [{'generated_text': [..., {'role': 'assistant', 'content': '...'}]}]
        assistant_response = outputs[0]["generated_text"][-1]
        
        # Extract the content from the assistant's message
        if assistant_response.get("role") == "assistant":
            content = assistant_response.get("content", "")
            return content.strip()
        else:
            logger.warning(f"Unexpected LLM output format. Expected 'assistant' role, got: {assistant_response}")
            # Fallback for unexpected formats, e.g., if the output is just a string
            if isinstance(outputs[0]["generated_text"], str):
                 return outputs[0]["generated_text"].strip()
            return str(outputs[0]["generated_text"]).strip()


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
        self.use_embeddings = use_embeddings
        self.llm = None
        self.embeddings = None
        self.pdf_files = []
        self.pdf_content_cache = {}
    
    # MODIFIED: Updated to use GptOssLLM
    def setup_llm(self):
        logger.info("Setting up GPT-OSS-120B...")
        self.llm = GptOssLLM(model_name="openai/gpt-oss-120b")
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
    
    def get_pdf_content(self, pdf_path: Path) -> str:
        """
        Load PDF content with caching to avoid loading same file multiple times.
        """
        if pdf_path in self.pdf_content_cache:
            return self.pdf_content_cache[pdf_path]
        
        logger.debug(f"Loading {pdf_path.name} from disk (caching for reuse)...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        full_text = "\n\n".join([page.page_content for page in pages]) if pages else ""
        self.pdf_content_cache[pdf_path] = full_text
        return full_text
    
    def extract_topics_from_pdf(self, pdf_path: Path) -> List[str]:
        """Extract topics from a single PDF (using cached content)."""
        logger.info(f"Extracting topics from: {pdf_path.name}")
        full_text = self.get_pdf_content(pdf_path)
        
        if not full_text:
            logger.warning(f"No content in {pdf_path.name}")
            return []
        
        prompt_text = full_text[:10000]
        
        prompt = f"""Analyze this weekly report and identify the main topics covered.

Report:
{prompt_text}

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
        
        all_topics = [topic for topics in all_topics_by_pdf.values() for topic in topics]
        logger.info(f"Total topics before consolidation: {len(all_topics)}")
        
        unique_topics = list(dict.fromkeys(all_topics))
        logger.info(f"After deduplication: {len(unique_topics)} topics")
        
        if len(unique_topics) <= self.max_topics:
            return unique_topics
        
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
            logger.warning(f"Using top {self.max_topics} from unique list as consolidation failed.")
            return unique_topics[:self.max_topics]
        
        return consolidated[:self.max_topics]
    
    def filter_relevant_content(self, full_text: str, topics: List[str]) -> str:
        """
        Use embeddings to filter PDF content to only relevant sections.
        """
        if not self.use_embeddings or not self.embeddings:
            return full_text[:12000]
        
        chunks = [full_text[i:i+800] for i in range(0, len(full_text), 700)]
        
        if len(chunks) <= 10:
            return full_text
        
        chunk_embeddings = self.embeddings.embed_documents(chunks)
        topic_embedding = self.embeddings.embed_query(" ".join(topics))
        
        similarities = np.dot(chunk_embeddings, topic_embedding)
        top_indices = np.argsort(similarities)[-15:][::-1]
        
        filtered_chunks = [chunks[i] for i in sorted(top_indices)]
        filtered_text = "\n\n".join(filtered_chunks)
        
        logger.info(f"  Filtered {len(chunks)} chunks → {len(filtered_chunks)} relevant chunks")
        return filtered_text
    
    def extract_all_topics_from_pdf(self, pdf_path: Path, topics: List[str]) -> Dict[str, List[str]]:
        """
        OPTIMIZED: Extract information about ALL topics from ONE PDF in a single LLM call.
        """
        logger.info(f"Extracting all {len(topics)} topics from {pdf_path.name}...")
        full_text = self.get_pdf_content(pdf_path)
        
        if not full_text:
            return {topic: [] for topic in topics}
        
        filtered_text = self.filter_relevant_content(full_text, topics)
        topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics)])
        
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
        topic_facts = self._parse_multi_topic_output(result, topics, pdf_path.stem)
        
        total_facts = sum(len(facts) for facts in topic_facts.values())
        logger.info(f"  Extracted {total_facts} facts across {len(topics)} topics")
        
        return topic_facts
    
    def _parse_multi_topic_output(self, text: str, topics: List[str], week_name: str) -> Dict[str, List[str]]:
        """
        Parse output where multiple topics and their facts are in one response.
        """
        topic_facts = {topic: [] for topic in topics}
        current_topic = None
        
        for line in text.split('\n'):
            line = line.strip()
            
            if line.startswith('TOPIC:'):
                topic_name = line.replace('TOPIC:', '').strip()
                current_topic = next((t for t in topics if t.lower() in topic_name.lower() or topic_name.lower() in t.lower()), None)
            
            elif line.startswith('FACT:') and current_topic:
                fact = line.replace('FACT:', '').strip()
                if fact and len(fact) > 10 and fact.upper() != 'NONE':
                    topic_facts[current_topic].append(f"[{week_name}] {fact}")
        
        return topic_facts
    
    def extract_info_for_all_topics_optimized(self, topics: List[str]) -> Dict[str, List[str]]:
        """
        OPTIMIZED: Process all PDFs, extracting all topics from each.
        """
        logger.info(f"\nExtracting {len(topics)} topics from {len(self.pdf_files)} PDFs...")
        logger.info(f"This will make {len(self.pdf_files)} LLM calls (instead of {len(topics) * len(self.pdf_files)})")
        
        topic_info = defaultdict(list)
        
        for pdf_file in tqdm(self.pdf_files, desc="Processing PDFs"):
            pdf_topic_facts = self.extract_all_topics_from_pdf(pdf_file, topics)
            for topic, facts in pdf_topic_facts.items():
                if facts:
                    topic_info[topic].extend(facts)
        
        return dict(topic_info)
    
    def _parse_text_list(self, text: str, prefix: str = "TOPIC:") -> List[str]:
        """Parse items from text output."""
        items = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith(prefix):
                item = line.replace(prefix, '').strip()
            elif line.startswith('- ') or line.startswith('* '):
                item = line[2:].strip()
            elif re.match(r'^\d+[\.\)]\s+', line):
                item = re.sub(r'^\d+[\.\)]\s+', '', line).strip()
            else:
                continue
            
            if item and len(item) > 3:
                items.append(item)
        
        return items
    
    def generate_report(self, topic_info: Dict[str, List[str]], output_file: str = "Consolidated_Weekly_Report.docx"):
        """Generate final Word document."""
        logger.info(f"Generating report: {output_file}")
        
        doc = Document()
        doc.add_heading("Consolidated Weekly Reports - All Topics", 0).alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_heading("Executive Summary", 1)
        total_facts = sum(len(facts) for facts in topic_info.values())
        summary = f"This report consolidates {len(self.pdf_files)} weekly reports across {len(topic_info)} topics, extracting {total_facts} unique updates."
        doc.add_paragraph(summary)
        doc.add_paragraph()
        
        for topic, facts in topic_info.items():
            doc.add_heading(topic, 1)
            if facts:
                for fact in facts:
                    doc.add_paragraph(fact, style='List Bullet')
            else:
                doc.add_paragraph("No information found.", style='Body Text')
            doc.add_paragraph()
        
        doc.add_page_break()
        doc.add_heading("Report Metadata", 2)
        doc.add_paragraph(f"Source weekly reports: {len(self.pdf_files)}")
        doc.add_paragraph(f"Consolidated topics: {len(topic_info)}")
        doc.add_paragraph(f"Total facts extracted: {total_facts}")
        doc.add_paragraph(f"Content filtering with embeddings: {'Enabled' if self.use_embeddings else 'Disabled'}")
        
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
            logger.error("No PDFs to process. Exiting.")
            return
        
        # Phase 1 & 2: Topic Extraction and Consolidation
        logger.info("\n" + "="*70 + "\nPHASE 1 & 2: Topic Extraction and Consolidation\n" + "="*70)
        all_topics_by_pdf = {pdf.name: self.extract_topics_from_pdf(pdf) for pdf in self.pdf_files}
        consolidated_topics = self.consolidate_topics(all_topics_by_pdf)
        logger.info("\nCONSOLIDATED TOPICS:")
        for i, topic in enumerate(consolidated_topics, 1):
            logger.info(f"  {i}. {topic}")
        
        # Phase 3: Information Extraction (Optimized)
        logger.info("\n" + "="*70 + "\nPHASE 3: Extract info (OPTIMIZED - inverted loop)\n" + "="*70)
        topic_info = self.extract_info_for_all_topics_optimized(consolidated_topics)
        
        # Phase 4: Report Generation
        logger.info("\n" + "="*70 + "\nPHASE 4: Generate report\n" + "="*70)
        self.generate_report(topic_info)
        
        print("\n" + "="*70 + "\nCOMPLETE!\n" + "="*70)
        print(f"PDFs processed: {len(self.pdf_files)}")
        print(f"Topics consolidated: {len(consolidated_topics)}")
        print(f"Total facts extracted: {sum(len(f) for f in topic_info.values())}")
        if consolidated_topics:
             print(f"Speedup: Approx. {len(consolidated_topics)}x faster due to {len(self.pdf_files)} LLM calls instead of {len(self.pdf_files) * len(consolidated_topics)}.")
        print("="*70)


def main():
    source_dir = "./source_pdfs"
    
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"Created empty source directory at '{source_dir}'. Please add your PDF files there and run again.")
        return
    
    if not any(Path(source_dir).glob("*.pdf")):
        print(f"No PDF files found in '{source_dir}'. Please add your PDF files and run again.")
        return
    
    print("Starting Optimized Weekly Report Pipeline...")
    
    # Choose whether to use embeddings for content filtering.
    # It's slower but can yield higher quality results for large documents.
    # Set use_embeddings=False for maximum speed.
    pipeline = OptimizedWeeklyReportPipeline(source_dir=source_dir, use_embeddings=False)
    
    try:
        # Set use_flash_attention=True if you have a compatible GPU and libraries installed
        pipeline.run_pipeline(use_flash_attention=False)
    except Exception as e:
        logger.error(f"The pipeline failed with an error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
