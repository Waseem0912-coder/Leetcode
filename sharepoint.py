#!/usr/bin/env python3

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

try:
    from PyPDF2 import PdfReader
except:
    raise ImportError("Install: pip install PyPDF2")

try:
    from docx import Document
except:
    raise ImportError("Install: pip install python-docx")


# ---------------- SETTINGS ----------------
PDF_FOLDER = Path("pdf")
OUTPUT_FILE = Path("combined_report.docx")

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
INFERENCE_MODEL = "Qwen/Qwen3-1.7B"

USE_CUDA = torch.cuda.is_available()
EMBED_DTYPE = torch.float16 if USE_CUDA else torch.float32
TOP_K = 6
MAX_NEW_TOKENS = 900
TEMPERATURE = 0.0


# ---------------- UTILITIES ----------------
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def extract_text(pdf: Path) -> str:
    reader = PdfReader(str(pdf))
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except:
            out.append("")
    return "\n".join(out)


def detect_sections(text: str):
    lines = text.splitlines()
    headers = []
    pat = re.compile(r'^(?:\d+\.)|^[A-Z][A-Z\s]{3,}$|.*:\s*$')
    for i,l in enumerate(lines):
        if pat.match(l.strip()):
            headers.append(i)

    if not headers:
        return [("Full Document", text)]

    sections=[]
    for i, h in enumerate(headers):
        title = lines[h].strip()
        start = h+1
        end = headers[i+1] if i+1 < len(headers) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if not body: body = "(Empty Section)"
        sections.append((title, body))
    return sections


def chunk(text: str, n=400, overlap=50):
    words = text.split()
    if len(words)<=n:
        return [text]
    chunks=[]
    i=0
    while i<len(words):
        chunks.append(" ".join(words[i:i+n]))
        i+=n-overlap
    return chunks


class GlobalIndex:
    def __init__(self):
        self.vecs = []
        self.meta = []

    def add(self, vecs: Tensor, metas):
        self.vecs.append(vecs.cpu())
        self.meta.extend(metas)

    def finalize(self):
        if not self.vecs: return
        self.vecs = torch.cat(self.vecs, dim=0)
        self.vecs = F.normalize(self.vecs, p=2, dim=1)

    def search(self, q: Tensor, k=5):
        q = q.cpu()
        sims = (self.vecs @ q.T).squeeze(1)
        top = torch.topk(sims, k)
        return [(int(i), float(sims[i])) for i in top.indices]


def embed_texts(model, tok, texts):
    device = next(model.parameters()).device
    max_len = 8192
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        pooled = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled


def generate(model, tok, prompt):
    device = next(model.parameters()).device
    try:
        msg=[{"role":"user","content":prompt}]
        chat = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok([chat], return_tensors="pt").to(device)
    except:
        enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE>0),
            temperature=TEMPERATURE
        )
    out_ids = out[0][len(enc.input_ids[0]):]
    return tok.decode(out_ids, skip_special_tokens=True).strip()


# ---------------- MAIN ----------------
def main():
    pdfs = list(PDF_FOLDER.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in ./pdf")
        return

    print("Loading embedding model...")
    etok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, padding_side="left")
    emodel = AutoModel.from_pretrained(EMBEDDING_MODEL, torch_dtype=EMBED_DTYPE)
    if USE_CUDA: emodel = emodel.to("cuda")

    print("Loading inference model...")
    itok = AutoTokenizer.from_pretrained(INFERENCE_MODEL)
    imodel = AutoModelForCausalLM.from_pretrained(INFERENCE_MODEL, device_map="auto")

    index = GlobalIndex()
    doc_db = []  # (pdf, section title, chunk text)

    # Build Global Corpus
    print("Indexing PDFs...")
    for pdf in pdfs:
        text = extract_text(pdf)
        secs = detect_sections(text)
        for title, body in secs:
            chs = chunk(body)
            for c in chs:
                doc_db.append((pdf.name, title, c))

    # Embed all chunks
    all_texts = [c for _,_,c in doc_db]
    metas    = [{"pdf":pdf,"section":sec} for pdf,sec,_ in doc_db]
    batch=12
    for i in range(0,len(all_texts),batch):
        embs = embed_texts(emodel, etok, all_texts[i:i+batch])
        index.add(embs, metas[i:i+batch])
    index.finalize()

    # Generate Combined Report
    doc = Document()
    doc.add_heading("Combined Report", level=1)

    print("Generating summaries...")
    for pdf in pdfs:
        text = extract_text(pdf)
        secs = detect_sections(text)

        doc.add_heading(pdf.name, level=2)
        for title, body in secs:
            query = f"Summarize unique events/things reported in section '{title}', concise bullet style."
            q_emb = embed_texts(emodel, etok, [query])
            hits = index.search(q_emb, TOP_K)

            ctx = body[:1000]
            retrieved = "\n\n".join([doc_db[i][2][:800] for i,_ in hits])

            prompt = f"""
One-sentence task: Summarize unique events/items from this section in a structured report.

SECTION TITLE: {title}

SECTION CONTENT (excerpt):
{ctx}

RELEVANT CONTEXT:
{retrieved}

Now produce a clear bullet point summary of unique things that happened:
""".strip()

            summary = generate(imodel, itok, prompt)

            doc.add_heading(title, level=3)
            doc.add_paragraph(summary)

    doc.save(OUTPUT_FILE)
    print(f"\n✅ Combined report saved to: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
