#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_report.py
--------------
End-to-end, multi-pass RAG pipeline that:
1) Ingests & indexes a large directory of PDFs into Chroma (custom Qwen embeddings).
2) Uses a local 4-bit Qwen3-1.7B (transformers + bitsandbytes) wrapped in LangChain.
3) Pass 1: Dynamically discovers 5–7 main topics from a broad sample of the corpus.
4) Pass 2: For each topic, retrieves relevant chunks and extracts key bullet points (JSON list).
5) Pass 3: LLM-based consolidation to deduplicate/merge semantically identical bullets (JSON list).
6) Generates a single structured .docx report.

Strict constraints satisfied:
- Generator LLM: Qwen/Qwen3-1.7B via transformers, 4-bit bitsandbytes, wrapped by HuggingFacePipeline.
- Embeddings: Qwen/Qwen3-Embedding-0.6B via a CUSTOM LangChain Embeddings class.
- Anti-hallucination: Prompts *explicitly* force “ONLY use provided context” and return [] if nothing found.

Usage:
    python auto_report.py --source ./source_pdfs --out Consolidated_Report.docx
"""

import os
import sys
import json
import argparse
import random
import warnings
from typing import List, Dict, Any, Iterable, Optional

import torch
import torch.nn.functional as F

# --- Transformers / HF
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline as hf_pipeline,
)

# --- LangChain core
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline

# --- Vector store & loaders
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# --- DOCX
from docx import Document
from docx.shared import Pt

# --- Misc
from tqdm import tqdm


# =========================
# Custom Qwen Embeddings
# =========================

class CustomQwenEmbeddings(Embeddings):
    """
    Custom embeddings using Qwen/Qwen3-Embedding-0.6B.
    Implements:
      - last_token_pool
      - get_detailed_instruct
      - embed_documents (batched; normalized)
      - embed_query (instruction reformatted; normalized)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        max_length: int = 1024,
        batch_size: int = 16,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        if torch_dtype is None:
            # Prefer bfloat16 if supported on GPU; otherwise float16 on GPU; else float32 on CPU
            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        # Qwen embedding model loads with trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pools by taking the hidden state of the last non-padding token per sequence.
        last_hidden_states: [B, L, H]
        attention_mask:     [B, L]
        returns:            [B, H]
        """
        # length per sequence = sum(attention_mask)
        lengths = attention_mask.sum(dim=1)  # [B]
        # index of last non-pad token: lengths - 1 (clamp >= 0)
        idx = torch.clamp(lengths - 1, min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, last_hidden_states.size(-1))
        # Gather last token hidden states
        gathered = last_hidden_states.gather(dim=1, index=idx).squeeze(1)  # [B, H]
        return gathered

    @staticmethod
    def get_detailed_instruct(query: str) -> str:
        """
        Formats the query with a generic retrieval instruction suitable for embedding tasks.
        """
        task_description = "Given a web search query, retrieve relevant passages that answer the query."
        return f"{task_description}\nQuery: {query}\n"

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Internal batching + forward pass + last-token pooling + L2 normalization.
        """
        all_vecs: List[List[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**enc)
                # Some embedding models expose different attr; fall back to last_hidden_state
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    hidden = outputs[0]

                pooled = self.last_token_pool(hidden, enc["attention_mask"])  # [B, H]
                pooled = F.normalize(pooled, p=2, dim=1)
                all_vecs.extend(pooled.detach().cpu().tolist())
        return all_vecs

    # ---- LangChain required methods ----
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        instruct = self.get_detailed_instruct(text)
        vec = self._embed_texts([instruct])[0]
        return vec


# =========================
# LLM Helpers
# =========================

def build_qwen_generator_4bit(
    model_name: str = "Qwen/Qwen3-1.7B",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    device_map: str = "auto",
) -> HuggingFacePipeline:
    """
    Loads Qwen/Qwen3-1.7B in 4-bit (nf4) with bitsandbytes, wraps in HF pipeline,
    then wraps in LangChain's HuggingFacePipeline.
    """
    try:
        _ = BitsAndBytesConfig
    except Exception as e:
        raise RuntimeError(
            "bitsandbytes is required to load Qwen3-1.7B in 4-bit. "
            "Please `pip install bitsandbytes` and ensure a compatible GPU."
        ) from e

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.eval()

    gen_pipe = hf_pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        # Default gen params; small model, JSON-only outputs; keep conservative.
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


def safe_parse_json_list(text: str) -> List[str]:
    """
    Robustly parse a JSON array from an LLM response.
    Extracts the first [...] block and json.loads it. Returns [] on failure.
    """
    try:
        # Find the first JSON list in the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1].strip()
            data = json.loads(snippet)
            if isinstance(data, list):
                # Ensure list of strings
                return [str(x).strip() for x in data if isinstance(x, (str, int, float))]
    except Exception:
        pass
    return []


def join_context(docs: List[Any], max_chars: int = 12000) -> str:
    """
    Join retrieved documents' page_content into a single context string with char cap.
    """
    parts = []
    total = 0
    for d in docs:
        t = d.page_content if hasattr(d, "page_content") else str(d)
        if not t:
            continue
        if total + len(t) > max_chars:
            t = t[: max_chars - total]
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(parts)


# =========================
# DOCX Generation
# =========================

def write_docx(compiled_data: Dict[str, List[str]], out_path: str) -> None:
    doc = Document()
    # Title
    title = doc.add_heading('Consolidated Report', 0)
    # Subtitle style for consistency
    p = doc.add_paragraph()
    run = p.add_run("Generated by auto_report.py")
    run.italic = True
    run.font.size = Pt(10)

    for topic, bullets in compiled_data.items():
        doc.add_heading(topic, level=1)
        if not bullets:
            para = doc.add_paragraph()
            para.add_run("(No relevant information found.)").italic = True
            continue
        for item in bullets:
            para = doc.add_paragraph(style='List Bullet')
            para.add_run(item)

    doc.save(out_path)


# =========================
# Main Pipeline
# =========================

def main(args: argparse.Namespace) -> None:
    random.seed(42)

    source_dir = args.source
    out_docx = args.out

    if not os.path.isdir(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- Step 1: Load PDFs
    print("Step 1: Loading PDFs ...")
    loader = PyPDFDirectoryLoader(source_dir, glob="**/*.pdf")
    raw_docs = loader.load()
    if len(raw_docs) == 0:
        print(f"[ERROR] No PDFs found under {source_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- Step 2: Chunking & Indexing
    print("Step 2: Chunking documents ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
        is_separator_regex=False,
    )
    docs = splitter.split_documents(raw_docs)
    print(f"  Total chunks: {len(docs)}")

    print("Step 2: Building custom Qwen embeddings ...")
    embeddings = CustomQwenEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        max_length=1024,
        batch_size=16,
    )

    print("Step 2: Creating Chroma vector store ...")
    # Memory-only store; for persistence, set persist_directory=...
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    # ---- Step 3: LLM (Qwen3-1.7B 4-bit) & Chains
    print("Step 3: Loading Qwen/Qwen3-1.7B (4-bit) ...")
    llm = build_qwen_generator_4bit(
        model_name="Qwen/Qwen3-1.7B",
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})  # k per topic

    # ---- Step 4: Dynamic Topic Generation (Pass 1)
    print("Step 4: Generating dynamic topics ...")
    # Broad sample: use a generic query to fetch diverse chunks, then feed to topic prompt
    broad_docs = retriever.get_relevant_documents("overall summary of main topics across the corpus")
    # If that's too narrow, add some random chunks as backup context
    if len(broad_docs) < 20:
        random_sample = random.sample(docs, min(40, len(docs)))
        broad_docs = random_sample + broad_docs

    context_text = join_context(broad_docs, max_chars=12000)

    topic_prompt = PromptTemplate.from_template(
        (
            "You are a careful analyst.\n"
            "Analyze ONLY the following context and identify the 5–7 most important, high-level topics.\n"
            "Base your answer ONLY on the provided context.\n\n"
            "CONTEXT:\n{context}\n\n"
            "Respond ONLY with a JSON list of strings. Example:\n"
            "['Project Budget', 'Security Risks', 'Timeline']"
        )
    )

    topic_chain = (
        {"context": RunnablePassthrough()}
        | topic_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(safe_parse_json_list)
    )

    dynamic_topics_list: List[str] = topic_chain.invoke(context_text)
    # Fallback if empty: propose generic buckets
    if not dynamic_topics_list:
        warnings.warn("Topic discovery returned empty list; falling back to generic topics.")
        dynamic_topics_list = [
            "Project Scope", "Timeline", "Budget & Costs", "Risks & Issues",
            "Architecture & Design", "Security & Compliance", "Dependencies"
        ]

    print(f"  Topics discovered: {dynamic_topics_list}")

    # ---- Step 5: Topic-Based Extraction Loop (Pass 2 & 3)
    print("Step 5: Extracting & deduplicating bullet points per topic ...")
    compiled_data: Dict[str, List[str]] = {}

    extract_prompt = PromptTemplate.from_template(
        (
            "You extract facts as bullet points.\n"
            "Base your answer ONLY on the provided context.\n"
            "If no relevant information is found, respond ONLY with [].\n\n"
            "TOPIC: {topic}\n\n"
            "CONTEXT:\n{context}\n\n"
            "Respond ONLY with a JSON list of brief, standalone bullet points (strings)."
        )
    )

    dedupe_prompt = PromptTemplate.from_template(
        (
            "You are a precise editor.\n"
            "Review the following list of bullet points and consolidate semantically identical items.\n"
            "Keep them short and non-overlapping. Return ONLY the final, unique JSON list of strings.\n\n"
            "ITEMS:\n{items}\n"
        )
    )

    # Dedicated chains for Pass 2 and Pass 3
    extract_chain = (
        {"topic": lambda x: x["topic"], "context": lambda x: x["context"]}
        | extract_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(safe_parse_json_list)
    )

    dedupe_chain = (
        {"items": RunnablePassthrough()}
        | dedupe_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(safe_parse_json_list)
    )

    for topic in tqdm(dynamic_topics_list, desc="Topics"):
        # A) Retrieve
        retrieved_docs = retriever.get_relevant_documents(topic)
        ctx = join_context(retrieved_docs, max_chars=10000)

        # B) Extract
        raw_items: List[str] = extract_chain.invoke({"topic": topic, "context": ctx})

        # If model returns empty, store [] immediately
        if not raw_items:
            compiled_data[topic] = []
            continue

        # Light local normalization before LLM dedupe
        normalized = []
        for r in raw_items:
            s = str(r).strip()
            # Clean trailing punctuation & whitespace
            if s.endswith((".", ";", ",")):
                s = s[:-1].strip()
            if s:
                normalized.append(s)

        # C) LLM Deduplication / Consolidation
        final_items = dedupe_chain.invoke(json.dumps(normalized, ensure_ascii=False))

        # Fallback: if empty after LLM dedupe, at least unique set locally
        if not final_items:
            final_items = sorted(set(normalized))

        compiled_data[topic] = final_items

    # ---- Step 6: DOCX Generation
    print(f"Step 6: Writing DOCX -> {out_docx}")
    write_docx(compiled_data, out_docx)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a consolidated .docx report from a PDF directory using a multi-pass RAG pipeline.")
    parser.add_argument("--source", type=str, default="./source_pdfs", help="Directory containing PDFs (recursively).")
    parser.add_argument("--out", type=str, default="Consolidated_Report.docx", help="Output DOCX filename.")
    args = parser.parse_args()
    main(args)
    
