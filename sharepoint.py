#!/usr/bin/env python3
"""
rag_pdf_reports.py

RAG-style pipeline over PDF files in ./pdf/
 - uses one model for embeddings
 - uses another model for generation/inference
 - creates a .docx report per PDF summarizing "unique things that happened" per section

Requirements:
 - Python 3.8+
 - transformers >= 4.51.0
 - torch
 - PyPDF2 (pip install PyPDF2)
 - python-docx (pip install python-docx)
 - (optional) faiss (pip install faiss-cpu) for faster retrieval

Notes:
 - Models, device usage, and precision are configurable below.
 - The script tries to load models in lower precision / on GPU when available.
"""

import os
import re
import math
import json
from typing import List, Tuple, Dict, Any
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# PDF extraction
try:
    from PyPDF2 import PdfReader
except Exception as e:
    raise ImportError("PyPDF2 is required. pip install PyPDF2") from e

# docx output
try:
    from docx import Document
except Exception as e:
    raise ImportError("python-docx is required. pip install python-docx") from e

# Optional faiss for vector index
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# ---------- User-configurable settings ----------
PDF_FOLDER = Path("pdf")
OUTPUT_FOLDER = Path("reports")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Embedding model (smaller) and inference model (larger)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
INFERENCE_MODEL = "Qwen/Qwen3-1.7B"

# Device and dtype preferences
USE_CUDA = torch.cuda.is_available()
EMBED_DTYPE = torch.float16 if USE_CUDA else torch.float32
INFER_DTYPE = "auto"  # pass through to from_pretrained; we select device_map="auto" for inference

# Chunking
CHUNK_SIZE_TOKENS = 1024  # approx words-based chunk size
CHUNK_OVERLAP = 128

# Retrieval
TOP_K = 6

# Generation params
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0

# -------------------------------------------------

# Utility: last_token_pool from the user's snippet to pool embeddings
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # Determine if tokenizer used left padding by checking last column of attention mask
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


# ---------------- PDF/Text utilities ----------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            # fallback: skip page if extraction fails
            pages.append("")
    return "\n\n".join(pages)


def simple_split_into_paragraphs(text: str) -> List[str]:
    # Normalize newlines and split on double newlines
    text = re.sub(r'\r\n?', '\n', text)
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paras


def detect_sections(text: str) -> List[Tuple[str, str]]:
    """
    Heuristic section detection:
     - find lines that look like headings: short (<120 chars), often all-caps or start with number + dot, or end with ':'
     - split by those headings; return list of (heading, body)
    Fallback: if no headings found, return one section with title = filename-like.
    """
    lines = text.splitlines()
    header_indices = []
    header_pattern = re.compile(r'^\s*(?:\d+\.)+\s+|^[A-Z0-9][A-Z0-9 \-]{3,}\s*$|.*:\s*$')
    for i, ln in enumerate(lines):
        if len(ln.strip()) == 0:
            continue
        if len(ln.strip()) < 120 and header_pattern.match(ln.strip()):
            header_indices.append(i)

    if not header_indices:
        # fallback: treat first 1000 chars as intro heading
        return [("Full Document", text)]

    sections = []
    for idx_i, hi in enumerate(header_indices):
        title = lines[hi].strip()
        start = hi + 1
        end = header_indices[idx_i + 1] if idx_i + 1 < len(header_indices) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if not body:
            # maybe heading was followed by heading; try to grab some following lines
            j = start
            snippet = []
            while j < len(lines) and len(snippet) < 10:
                snippet.append(lines[j])
                j += 1
            body = "\n".join(snippet).strip()
        sections.append((title, body))
    return sections


def word_tokenizer_simple(text: str) -> List[str]:
    # simple whitespace tokenizer for chunking heuristics
    return text.split()


def chunk_text(text: str, chunk_size_words: int = 400, overlap_words: int = 50) -> List[str]:
    words = word_tokenizer_simple(text)
    if len(words) <= chunk_size_words:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size_words]
        chunks.append(" ".join(chunk))
        i += (chunk_size_words - overlap_words)
    return chunks


# ---------------- Embedding + Index ----------------
class SimpleVectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.embs = None  # numpy array (N, dim)
        self.meta = []  # list of metadata dicts (source pdf, section, chunk_text)
        self.ids = []

    def add(self, vectors: Tensor, metas: List[Dict[str, Any]]):
        # vectors: torch tensor CPU float32/float16
        import numpy as np
        vecs = vectors.detach().cpu().numpy()
        if self.embs is None:
            self.embs = vecs
            self.meta = metas.copy()
        else:
            self.embs = np.vstack([self.embs, vecs])
            self.meta.extend(metas)

    def search(self, query_vec: Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        # Returns list of (index, score) sorted desc
        import numpy as np
        q = query_vec.detach().cpu().numpy().astype(self.embs.dtype)
        # cosine similarity: normalized inputs expected
        sims = (self.embs @ q.T).squeeze(-1)
        if sims.ndim == 0:
            sims = np.array([sims])
        topk_idx = sims.argsort()[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in topk_idx]


def build_index_from_texts(embedding_model, tokenizer, texts: List[str], metas: List[Dict[str, Any]], batch_size: int = 8, device=None):
    """
    Returns vector index and the normalized embeddings tensor
    """
    all_embs = []
    model_device = device if device is not None else next(embedding_model.parameters()).device
    max_len = 8192
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(model_device) for k, v in enc.items()}
        with torch.no_grad():
            out = embedding_model(**enc)
            pooled = last_token_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
        all_embs.append(pooled.cpu())
    all_embs_t = torch.cat(all_embs, dim=0)
    # create index
    idx = SimpleVectorIndex(dim=all_embs_t.shape[1])
    idx.add(all_embs_t, metas)
    return idx, all_embs_t


# ---------------- Inference / Generation ----------------
def generate_with_model(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0, device=None) -> str:
    """
    Attempt to use a chat template if tokenizer provides it (like Qwen),
    otherwise feed direct prompt and use generate.
    """
    model_device = device if device is not None else next(model.parameters()).device

    # If tokenizer has apply_chat_template (Qwen style), we can wrap message into chat format
    try:
        # Build chat-format input similar to user's snippet
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer([text], return_tensors="pt").to(model_device)
    except Exception:
        # fallback
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model_device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    # strip input tokens
    generated_ids = gen[0][len(inputs["input_ids"][0]):].tolist()
    out_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return out_text


# ---------------- Main pipeline ----------------
def process_pdf_file(pdf_path: Path,
                     embedding_tokenizer,
                     embedding_model,
                     infer_tokenizer,
                     infer_model,
                     vector_index: SimpleVectorIndex = None):
    print(f"\nProcessing {pdf_path} ...")
    text = extract_text_from_pdf(pdf_path)
    sections = detect_sections(text)
    # If vector_index is None, we will build an index from all chunks of this document only
    doc_chunks = []
    doc_metas = []
    for sec_idx, (title, body) in enumerate(sections):
        # break body into chunks
        chunks = chunk_text(body, chunk_size_words=CHUNK_SIZE_TOKENS // 2, overlap_words=CHUNK_OVERLAP)
        for c_idx, chunk in enumerate(chunks):
            meta = {
                "pdf": str(pdf_path.name),
                "section_idx": sec_idx,
                "section_title": title,
                "chunk_idx": c_idx,
            }
            doc_chunks.append(chunk)
            doc_metas.append(meta)

    # build index for this document (or use global index if provided)
    if vector_index is None:
        print("Building local index for the document ...")
        local_index, _ = build_index_from_texts(embedding_model, embedding_tokenizer, doc_chunks, doc_metas, batch_size=8, device=next(embedding_model.parameters()).device)
    else:
        local_index = vector_index
        # If using global, we need to add this doc's chunks to it
        # (caller might have already added them)

    # For each section, retrieve top-k relevant chunks, then call inference model to produce report
    doc_report = Document()
    doc_report.add_heading(f"Report for {pdf_path.name}", level=1)
    for sec_idx, (title, body) in enumerate(sections):
        doc_report.add_heading(f"Section: {title}", level=2)
        # create a short retrieval query from the section header + instruction
        task = "For the given document section, list all unique things/events/items that happened or were reported, organized in a concise report-like format with bullets and dates/times if present."
        short_instruction = "Create a concise report listing unique events/things that happened in the section."
        query_text = get_detailed_instruct(task, f"{title}\n\n{body[:400]}")  # include snippet as context

        # embed query
        enc = embedding_tokenizer([query_text], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(next(embedding_model.parameters()).device)
        with torch.no_grad():
            out = embedding_model(**enc)
            q_vec = last_token_pool(out.last_hidden_state, enc["attention_mask"])
            q_vec = F.normalize(q_vec, p=2, dim=1)

        # search
        hits = local_index.search(q_vec, top_k=TOP_K)
        retrieved_texts = []
        for idx, score in hits:
            meta = local_index.meta[idx]
            retrieved_texts.append({
                "meta": meta,
                "text": local_index.embs[idx].tolist()  # we won't include actual text here; instead we will index mapping
            })
        # Instead of storing text in index, we still have doc_chunks/doc_metas; match by meta
        matched_chunks = []
        for idx, score in hits:
            matched_chunks.append((doc_chunks[idx], doc_metas[idx], float(score)))

        # assemble context prompt: include instruction sentence + retrieved chunks (concise)
        context_parts = []
        context_parts.append(short_instruction)
        context_parts.append(f"Section title: {title}")
        context_parts.append("Section excerpt (first 1200 chars):\n" + body[:1200])
        context_parts.append("\nRetrieved relevant chunks (highest first):")
        for ctext, meta, sc in matched_chunks:
            context_parts.append(f"[score={sc:.4f}] Chunk (from {meta['pdf']} - {meta['section_title']}):\n{ctext[:1000]}")
        context_prompt = "\n\n".join(context_parts)
        # single-sentence instruction required per user earlier — include that at top
        one_sentence_instruction = "Summarize unique events/items in a compact report format (bullets, dates if any)."

        full_prompt = one_sentence_instruction + "\n\n" + context_prompt + "\n\nPlease produce a clear, short report for this section."

        # call inference model
        try:
            generated = generate_with_model(infer_model, infer_tokenizer, full_prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, device=next(infer_model.parameters()).device)
        except Exception as e:
            print("Generation failed:", e)
            generated = "ERROR: generation failed."

        # write to docx
        doc_report.add_paragraph(generated)

    # save docx
    out_path = OUTPUT_FOLDER / (pdf_path.stem + "_report.docx")
    doc_report.save(str(out_path))
    print(f"Saved report: {out_path}")
    return str(out_path)


def main():
    # Check pdf folder
    if not PDF_FOLDER.exists() or not PDF_FOLDER.is_dir():
        raise FileNotFoundError("Please create a folder named 'pdf' and put PDFs inside it.")

    # Load embedding model/tokenizer
    print("Loading embedding model and tokenizer ...")
    device_embed = "cuda" if USE_CUDA else "cpu"
    # Use low-precision and attn implementation where possible (user suggested flash_attention_2)
    try:
        embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, padding_side="left")
        embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL, torch_dtype=EMBED_DTYPE)
        if USE_CUDA:
            embedding_model = embedding_model.to("cuda")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model {EMBEDDING_MODEL}: {e}")

    # Load inference model/tokenizer
    print("Loading inference model and tokenizer ... (may take time)")
    try:
        infer_tokenizer = AutoTokenizer.from_pretrained(INFERENCE_MODEL)
        infer_model = AutoModelForCausalLM.from_pretrained(INFERENCE_MODEL, torch_dtype=None, device_map="auto")
        # optionally convert to 8-bit/4-bit quantized using bitsandbytes if available (not enforced here)
    except Exception as e:
        # fallback: attempt device cpu
        print("Warning: failed to load inference model with device_map=auto, trying CPU ...", e)
        infer_model = AutoModelForCausalLM.from_pretrained(INFERENCE_MODEL, torch_dtype=torch.float32)
        infer_tokenizer = AutoTokenizer.from_pretrained(INFERENCE_MODEL)

    # Prepare a global vector index optionally (if you want to index entire collection)
    # For simplicity we'll build per-document local index in process_pdf_file
    pdf_files = sorted([p for p in PDF_FOLDER.iterdir() if p.suffix.lower() in (".pdf",)])
    if not pdf_files:
        print("No PDFs found in ./pdf. Exiting.")
        return

    # Optionally try to warm up embedding model with a small call
    try:
        _ = embedding_tokenizer(["hello world"], return_tensors="pt")
    except Exception:
        pass

    # Process each PDF
    for pdf in pdf_files:
        try:
            process_pdf_file(pdf, embedding_tokenizer, embedding_model, infer_tokenizer, infer_model, vector_index=None)
        except Exception as e:
            print(f"Failed processing {pdf}: {e}")

    print("All done. Reports saved to:", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
