#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_report.py
--------------
RAG pipeline + optional strict parsing mode.

Key update: "Evidence-guarded" extraction (default ON)
- LLM must return [{"bullet": "...", "evidence": ["<verbatim-from-context>", ...]}, ...]
- We verify each bullet's evidence is present in the exact context; unsupported bullets are dropped.
- Paraphrase is OK, hallucination is not.
- Deterministic semantic dedup via Qwen embeddings (no LLM rewriting).

Usage:
  python auto_report.py --source ./source_pdfs --out Consolidated_Report.docx
  python auto_report.py --use-fixed-topics --topics "Budget, Risks"
  python auto_report.py --allow-unguarded    # disable evidence guard (not recommended)

Also includes:
  --strict-updates-mode  # deterministic Topic/Update regex parsing (no LLM)
"""

import os
import sys
import re
import json
import argparse
import random
import warnings
from typing import List, Dict, Any, Optional, Iterable, Tuple

import torch
import torch.nn.functional as F
import numpy as np

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
# Defaults & constants
# =========================

DEFAULT_TOPICS: List[str] = [
    "Project Scope",
    "Timeline",
    "Budget & Costs",
    "Risks & Issues",
    "Architecture & Design",
    "Security & Compliance",
    "Dependencies",
]

MIN_DOCS_PER_TOPIC = 6
MAX_CONTEXT_CHARS_TOPIC = 10000
MAX_CONTEXT_CHARS_TOPICS_DISCOVERY = 12000

# Strict mode (regex) defaults
DEFAULT_TOPIC_PATTERN = r'^\s*(Topic\s*[:\-]?\s*.+?)\s*$'
DEFAULT_UPDATE_PATTERN = r'^\s*(Update\s*.+?)\s*$'


# =========================
# Custom Qwen Embeddings
# =========================

class CustomQwenEmbeddings(Embeddings):
    """
    Custom embeddings using Qwen/Qwen3-Embedding-0.6B.
    Implements last_token_pool, get_detailed_instruct, normalized outputs.
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
            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

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
        lengths = attention_mask.sum(dim=1)
        idx = torch.clamp(lengths - 1, min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, last_hidden_states.size(-1))
        gathered = last_hidden_states.gather(dim=1, index=idx).squeeze(1)
        return gathered

    @staticmethod
    def get_detailed_instruct(query: str) -> str:
        task_description = "Given a web search query, retrieve relevant passages that answer the query."
        return f"{task_description}\nQuery: {query}\n"

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        all_vecs: List[List[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**enc)
                hidden = getattr(outputs, "last_hidden_state", outputs[0])
                pooled = self.last_token_pool(hidden, enc["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                all_vecs.extend(pooled.detach().cpu().tolist())
        return all_vecs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        instruct = self.get_detailed_instruct(text)
        vec = self._embed_texts([instruct])[0]
        return vec


# =========================
# LLM builder (Qwen3-1.7B 4-bit)
# =========================

def build_qwen_generator_4bit(
    model_name: str = "Qwen/Qwen3-1.7B",
    max_new_tokens: int = 384,
    temperature: float = 0.0,   # guardrail: deterministic
    top_p: float = 1.0,
    device_map: str = "auto",
) -> HuggingFacePipeline:
    # requires bitsandbytes
    _ = BitsAndBytesConfig  # raises if not importable
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
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=False,                # deterministic
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


# =========================
# Utilities
# =========================

def safe_parse_json_list(text: str) -> List[str]:
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1].strip()
            data = json.loads(snippet)
            if isinstance(data, list):
                return [str(x).strip() for x in data if isinstance(x, (str, int, float))]
    except Exception:
        pass
    return []


def safe_parse_json_objects(text: str) -> List[Dict[str, Any]]:
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1].strip()
            data = json.loads(snippet)
            if isinstance(data, list):
                out = []
                for d in data:
                    if isinstance(d, dict) and "bullet" in d and "evidence" in d:
                        ev = d.get("evidence") or []
                        ev = [str(x).strip() for x in ev if str(x).strip()]
                        out.append({"bullet": str(d["bullet"]).strip(), "evidence": ev})
                return out
    except Exception:
        pass
    return []


def join_context(docs: List[Any], max_chars: int) -> str:
    parts, total = [], 0
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


def retrieve_docs(retriever, query: str):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return []


def evidence_supported(context: str, evidence_list: List[str]) -> bool:
    """Return True if ANY evidence snippet appears verbatim (case-insensitive) in context."""
    if not evidence_list:
        return False
    ctx = context.lower()
    for ev in evidence_list:
        e = ev.strip().lower()
        if e and e in ctx:
            return True
    return False


def semantic_dedupe(bullets: List[str], embedder: CustomQwenEmbeddings, threshold: float = 0.92) -> List[str]:
    """Greedy clustering by cosine (vectors are already L2-normalized)."""
    if not bullets:
        return bullets
    vecs = np.array(embedder.embed_documents(bullets), dtype=np.float32)  # [n, d]
    keep_idx: List[int] = []
    for i in range(len(bullets)):
        v = vecs[i]
        if not keep_idx:
            keep_idx.append(i)
            continue
        sims = np.dot(vecs[keep_idx], v)
        if float(np.max(sims)) < threshold:
            keep_idx.append(i)
    return [bullets[i] for i in keep_idx]


# ---------- robust retrieval (to avoid empty sections)

def expand_queries_with_llm(llm: HuggingFacePipeline, topic: str, n: int = 4) -> List[str]:
    prompt = (
        "Generate concise, diverse retrieval queries that restate the TOPIC without adding facts.\n"
        f"TOPIC: {topic}\n"
        f"Return ONLY a JSON list of {n} short strings."
    )
    out = llm.invoke(prompt)
    variants = safe_parse_json_list(out)
    base = [topic.strip()]
    for v in variants:
        v = str(v).strip()
        if v and v.lower() != topic.strip().lower():
            base.append(v)
    # dedupe while preserving
    seen, result = set(), []
    for q in base:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            result.append(q)
    return result


def robust_retrieve_for_topic(
    vectorstore: Chroma,
    retriever,
    embeddings: CustomQwenEmbeddings,
    llm: HuggingFacePipeline,
    topic: str,
    k: int = 12,
    min_docs: int = MIN_DOCS_PER_TOPIC,
) -> List[Any]:
    collected: List[Any] = []
    doc_keys = set()

    def add_docs(docs: Iterable[Any]):
        for d in docs or []:
            md = getattr(d, "metadata", {}) or {}
            key = md.get("id") or f"{md.get('source')}::{md.get('page', '')}" or (getattr(d, "page_content", "")[:80])
            if key not in doc_keys:
                doc_keys.add(key)
                collected.append(d)

    # 1) baseline
    add_docs(retrieve_docs(retriever, topic))
    if len(collected) >= min_docs:
        return collected[:k]

    # 2) expanded queries
    for q in expand_queries_with_llm(llm, topic, n=4):
        if len(collected) >= k:
            break
        add_docs(retrieve_docs(retriever, q))
    if len(collected) >= min_docs:
        return collected[:k]

    # 3) direct vectorstore string search
    for q in expand_queries_with_llm(llm, topic, n=2):
        if len(collected) >= k:
            break
        try:
            add_docs(vectorstore.similarity_search(q, k=k))
        except Exception:
            pass
    if len(collected) >= min_docs:
        return collected[:k]

    # 4) vector-by-vector
    try:
        vec = embeddings.embed_query(topic)
        add_docs(vectorstore.similarity_search_by_vector(vec, k=k))
    except Exception:
        pass

    return collected[:k]


# =========================
# STRICT MODE parsing helpers
# =========================

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for s in items:
        k = s.strip()
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _split_multi_updates(line: str) -> List[str]:
    ids = re.findall(r'update\s+([A-Za-z0-9._\-]+)', line, flags=re.I)
    if len(ids) >= 2:
        return [f"Update {i}" for i in ids]
    return [line.strip()]


def extract_topics_updates_from_docs(
    raw_docs: List[Any],
    topic_pattern: str = DEFAULT_TOPIC_PATTERN,
    update_pattern: str = DEFAULT_UPDATE_PATTERN,
) -> Dict[str, List[str]]:
    topic_re = re.compile(topic_pattern, flags=re.IGNORECASE)
    update_re = re.compile(update_pattern, flags=re.IGNORECASE)
    compiled: Dict[str, List[str]] = {}

    for doc in raw_docs:
        text = getattr(doc, "page_content", "") or ""
        if not text.strip():
            continue
        current_topic: Optional[str] = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            tm = topic_re.match(line)
            if tm:
                current_topic = (tm.group(1) or tm.group(0)).strip()
                compiled.setdefault(current_topic, [])
                continue

            if current_topic:
                um = update_re.match(line)
                if um:
                    update_line = (um.group(1) or um.group(0)).strip()
                    for u in _split_multi_updates(update_line):
                        compiled[current_topic].append(u)

    for k in list(compiled.keys()):
        compiled[k] = _dedupe_preserve_order(compiled[k])
    return compiled


# =========================
# DOCX generation
# =========================

def write_docx(compiled_data: Dict[str, List[str]], out_path: str) -> None:
    doc = Document()
    doc.add_heading('Consolidated Report', 0)
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
# Main
# =========================

def parse_topics_arg(val: Optional[str]) -> Optional[List[str]]:
    if not val:
        return None
    try:
        if val.strip().startswith("["):
            data = json.loads(val)
            if isinstance(data, list):
                topics = [str(x).strip() for x in data if str(x).strip()]
                return list(dict.fromkeys(topics)) or None
    except Exception:
        pass
    topics = [t.strip() for t in val.split(",") if t.strip()]
    return list(dict.fromkeys(topics)) or None


def main(args: argparse.Namespace) -> None:
    random.seed(42)

    source_dir = args.source
    out_docx = args.out

    if not os.path.isdir(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    # Load PDFs (both modes)
    print("Step 1: Loading PDFs ...")
    loader = PyPDFDirectoryLoader(source_dir, glob="**/*.pdf")
    raw_docs = loader.load()
    if len(raw_docs) == 0:
        print(f"[ERROR] No PDFs found under {source_dir}", file=sys.stderr)
        sys.exit(1)

    # STRICT updates mode (no LLM)
    if args.strict_updates_mode:
        print("STRICT mode: parsing Topic/Update verbatim ...")
        compiled_all = extract_topics_updates_from_docs(
            raw_docs,
            topic_pattern=args.topic_pattern or DEFAULT_TOPIC_PATTERN,
            update_pattern=args.update_pattern or DEFAULT_UPDATE_PATTERN,
        )
        if args.use_fixed_topics:
            user_topics = parse_topics_arg(args.topics)
            topics_to_emit = user_topics if user_topics else DEFAULT_TOPICS
            compiled_data = {t: compiled_all.get(t, []) for t in topics_to_emit}
        else:
            compiled_data = compiled_all

        print(f"Writing DOCX -> {out_docx}")
        write_docx(compiled_data, out_docx)
        print("Done.")
        return

    # -----------------------
    # RAG mode (LLM-powered)
    # -----------------------
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

    print("Step 3: Building custom Qwen embeddings + Chroma ...")
    embeddings = CustomQwenEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        max_length=1024,
        batch_size=16,
    )
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    print("Step 4: Loading Qwen/Qwen3-1.7B (4-bit) ...")
    llm = build_qwen_generator_4bit(
        model_name="Qwen/Qwen3-1.7B",
        max_new_tokens=384,
        temperature=0.0 if not args.temperature else float(args.temperature),
        top_p=1.0,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    # Topics list
    use_fixed = bool(args.use_fixed_topics)
    user_topics = parse_topics_arg(args.topics)
    if use_fixed:
        if user_topics:
            dynamic_topics_list: List[str] = user_topics
            print(f"Step 5: Using FIXED topics from --topics: {dynamic_topics_list}")
        else:
            dynamic_topics_list = DEFAULT_TOPICS
            print(f"Step 5: Using FIXED built-in topics: {dynamic_topics_list}")
    else:
        print("Step 5: Generating dynamic topics ...")
        broad_docs = retrieve_docs(retriever, "overall summary of main topics across the corpus")
        if len(broad_docs) < 20:
            random_sample = random.sample(docs, min(40, len(docs)))
            broad_docs = random_sample + (broad_docs or [])
        context_text = join_context(broad_docs, max_chars=MAX_CONTEXT_CHARS_TOPICS_DISCOVERY)

        topic_prompt = PromptTemplate.from_template(
            "You are a careful analyst.\n"
            "Analyze ONLY the following context and identify the 5–7 most important, high-level topics.\n"
            "Base your answer ONLY on the provided context.\n\n"
            "CONTEXT:\n{context}\n\n"
            "Respond ONLY with a JSON list of strings. Example:\n"
            "['Project Budget', 'Security Risks', 'Timeline']"
        )

        topic_chain = (
            {"context": RunnablePassthrough()}
            | topic_prompt
            | llm
            | StrOutputParser()
            | RunnableLambda(safe_parse_json_list)
        )

        dynamic_topics_list = topic_chain.invoke(context_text)
        if not dynamic_topics_list:
            warnings.warn("Topic discovery returned empty list; falling back to built-in defaults.")
            dynamic_topics_list = DEFAULT_TOPICS

        print(f"  Topics discovered: {dynamic_topics_list}")

    # Evidence-guarded extraction
    print("Step 6: Evidence-guarded extraction & semantic dedup ...")
    compiled_data: Dict[str, List[str]] = {}

    extract_prompt_guarded = PromptTemplate.from_template(
        (
            "You extract grounded bullet points.\n"
            "Base your answer ONLY on the provided context. Do NOT invent facts.\n"
            "If no relevant information is found, respond ONLY with [].\n\n"
            "TOPIC: {topic}\n\n"
            "CONTEXT:\n{context}\n\n"
            "Return ONLY a JSON list of objects. Each object MUST have:\n"
            "- 'bullet': a short standalone paraphrase grounded in the context\n"
            "- 'evidence': a list of 1–3 short quotes COPIED VERBATIM from the context that justify the bullet\n\n"
            "Example:\n"
            "[{\"bullet\": \"Budget increased by 10% in Q4.\", \"evidence\": [\"budget increased by 10% in Q4\"]}]\n"
        )
    )

    extract_guarded_chain = (
        {"topic": lambda x: x["topic"], "context": lambda x: x["context"]}
        | extract_prompt_guarded
        | llm
        | StrOutputParser()
        | RunnableLambda(safe_parse_json_objects)
    )

    for topic in tqdm(dynamic_topics_list, desc="Topics"):
        # robust retrieval to avoid empty sections
        retrieved_docs = robust_retrieve_for_topic(
            vectorstore=vectorstore,
            retriever=retriever,
            embeddings=embeddings,
            llm=llm,
            topic=topic,
            k=12,
            min_docs=MIN_DOCS_PER_TOPIC,
        )
        ctx = join_context(retrieved_docs, max_chars=MAX_CONTEXT_CHARS_TOPIC)

        if args.allow_unguarded:
            # fallback to simpler list extraction (no evidence verification)
            simple_prompt = PromptTemplate.from_template(
                "Base your answer ONLY on the provided context.\n"
                "If no relevant information is found, respond ONLY with [].\n\n"
                "TOPIC: {topic}\n\n"
                "CONTEXT:\n{context}\n\n"
                "Respond ONLY with a JSON list of brief, standalone bullet points (strings)."
            )
            simple_chain = (
                {"topic": lambda x: x["topic"], "context": lambda x: x["context"]}
                | simple_prompt
                | llm
                | StrOutputParser()
                | RunnableLambda(safe_parse_json_list)
            )
            bullets = simple_chain.invoke({"topic": topic, "context": ctx})
            bullets = [b.strip() for b in bullets if b and b.strip()]
            bullets = semantic_dedupe(bullets, embeddings, threshold=0.92)
            compiled_data[topic] = bullets
            print(f"[Topic] {topic} -> docs={len(retrieved_docs)}, bullets={len(bullets)} (unguarded)")
            continue

        # guarded path
        objs = extract_guarded_chain.invoke({"topic": topic, "context": ctx})
        kept: List[str] = []
        for obj in objs:
            bullet = (obj.get("bullet") or "").strip()
            ev = obj.get("evidence") or []
            if not bullet:
                continue
            # verify at least one evidence quote appears in context
            if evidence_supported(ctx, ev):
                kept.append(bullet)

        # deterministic semantic dedup (no LLM rewriting)
        kept = semantic_dedupe(kept, embeddings, threshold=0.92)

        compiled_data[topic] = kept
        print(f"[Topic] {topic} -> docs={len(retrieved_docs)}, bullets_kept={len(kept)}")

    print(f"Step 7: Writing DOCX -> {out_docx}")
    write_docx(compiled_data, out_docx)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a consolidated .docx report from a PDF directory.")
    parser.add_argument("--source", type=str, default="./source_pdfs", help="Directory containing PDFs (recursively).")
    parser.add_argument("--out", type=str, default="Consolidated_Report.docx", help="Output DOCX filename.")

    # Strict deterministic extraction flags
    parser.add_argument("--strict-updates-mode", action="store_true",
                        help="Deterministically parse 'Topic ...' headers and 'Update ...' lines and merge them verbatim (no LLM).")
    parser.add_argument("--topic-pattern", type=str, default=None,
                        help="Regex for topic headers (case-insensitive). Default matches 'Topic ...'.")
    parser.add_argument("--update-pattern", type=str, default=None,
                        help="Regex for update lines (case-insensitive). Default matches 'Update ...'.")

    # RAG topic control
    parser.add_argument("--use-fixed-topics", action="store_true",
                        help="Use a fixed list of topics instead of dynamic topic discovery.")
    parser.add_argument("--topics", type=str, default=None,
                        help="Comma-separated or JSON array of topics when --use-fixed-topics is set.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override generator temperature (default 0.0).")

    # Guardrail
    parser.add_argument("--allow-unguarded", action="store_true",
                        help="Disable evidence guard (not recommended).")

    args = parser.parse_args()
    main(args)
