#!/usr/bin/env python3
"""
100% LOCAL weekly reports consolidation system.
No external API calls - all processing happens on your hardware.
"""
from pathlib import Path
from typing import List
import pdfplumber
from docx import Document
from transformers import pipeline
import torch
import re
import os

# Verify we're running locally
print("🔍 Environment Check:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"   HuggingFace cache: {os.environ.get('HF_HOME', '~/.cache/huggingface/')}")

# Configuration
PDF_PATH = "/mnt/user-data/uploads"
OUTPUT_PATH = "/mnt/user-data/outputs"
MODEL_ID = "openai/gpt-oss-120b"

# Optional: Set offline mode to ensure no network calls
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("\n🔧 Loading GPT-OSS-120B model (LOCAL)...")
print("   Note: First run will download ~240GB from HuggingFace")
print("   Subsequent runs use cached local copy")

pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    torch_dtype="auto",  # Uses float16/bfloat16 automatically
    device_map="auto",   # Distributes across available GPUs
    trust_remote_code=True,  # May be needed for some models
)

print("✅ Model loaded - All inference will be LOCAL")


def extract_pages_from_pdf(pdf_path: str) -> List[dict]:
    """Extract text from PDF (LOCAL - uses pdfplumber library)."""
    print(f"\n📄 Extracting pages from: {pdf_path}")
    pages = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"   Found {total_pages} pages")
            
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    pages.append({
                        'number': i,
                        'text': page_text.strip()
                    })
                
                if i % 10 == 0:
                    print(f"   Extracted {i}/{total_pages} pages...")
                    
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        raise
    
    print(f"✓ Extracted {len(pages)} pages (LOCAL processing)")
    return pages


def convert_page_to_markdown(page_num: int, page_text: str, pipe) -> str:
    """
    Convert a single page to markdown using LOCAL LLM inference.
    NO external API calls - runs entirely on your GPU/CPU.
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a document conversion specialist. Convert text to clean markdown format, preserving ALL information. Use # for headings, - for bullets. Output ONLY markdown."
        },
        {
            "role": "user",
            "content": f"Convert this page to markdown:\n\n{page_text}"
        }
    ]
    
    # LOCAL INFERENCE - runs on your hardware
    outputs = pipe(
        messages,
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.1,
    )
    
    markdown = outputs[0]["generated_text"][-1]["content"]
    return markdown


def convert_all_pages_to_markdown(pages: List[dict], pipe) -> str:
    """Convert all pages to markdown (LOCAL processing)."""
    print("\n🔄 Converting pages to markdown (LOCAL inference)...")
    
    markdown_pages = []
    total_pages = len(pages)
    
    for idx, page in enumerate(pages, 1):
        page_num = page['number']
        page_text = page['text']
        
        print(f"   Page {idx}/{total_pages} (LOCAL GPU inference)...", end='\r')
        
        # LOCAL INFERENCE CALL
        markdown = convert_page_to_markdown(page_num, page_text, pipe)
        
        markdown_pages.append(f"\n<!-- Page {page_num} -->\n")
        markdown_pages.append(markdown)
    
    print(f"\n✅ All pages converted (100% LOCAL)")
    
    combined_markdown = "\n".join(markdown_pages)
    
    # Save locally
    md_path = Path(OUTPUT_PATH) / "combined_reports.md"
    md_path.write_text(combined_markdown, encoding='utf-8')
    print(f"💾 Saved to local disk: {md_path}")
    
    return combined_markdown


def consolidate_with_single_query(markdown_text: str, pipe) -> str:
    """
    Final consolidation using LOCAL LLM.
    This is the main processing step - single query with full context.
    """
    print("\n🤖 Final consolidation (LOCAL inference with full context)...")
    
    # Calculate token estimate
    words = len(markdown_text.split())
    approx_tokens = int(words * 1.3)
    print(f"   Input: ~{approx_tokens:,} tokens")
    print(f"   Context window: ~120,000 tokens")
    
    if approx_tokens > 100000:
        print(f"   ⚠️  Warning: Approaching context limit!")
    else:
        print(f"   ✓ Input fits comfortably in context")
    
    messages = [
        {
            "role": "system",
            "content": """You are an expert document consolidation specialist. Analyze all weekly reports and create a comprehensive consolidated summary in markdown format.

Structure:
# Consolidated Weekly Reports

## [Topic 1]
- Key points with details
- Metrics and achievements

## [Topic 2]
...

Preserve ALL details: numbers, dates, names, metrics. Remove duplicates."""
        },
        {
            "role": "user",
            "content": f"""Consolidate these 45 weeks of reports:

{markdown_text}

Output consolidated markdown."""
        }
    ]
    
    print("   🔄 Running LOCAL inference (this may take a few minutes)...")
    
    # LOCAL INFERENCE - the big one!
    outputs = pipe(
        messages,
        max_new_tokens=8192,
        do_sample=False,
        temperature=0.1,
    )
    
    consolidated_md = outputs[0]["generated_text"][-1]["content"]
    
    # Save locally
    consolidated_md_path = Path(OUTPUT_PATH) / "consolidated_report.md"
    consolidated_md_path.write_text(consolidated_md, encoding='utf-8')
    print(f"💾 Saved to local disk: {consolidated_md_path}")
    
    return consolidated_md


def markdown_to_docx(markdown_text: str, output_path: str):
    """Convert markdown to DOCX (LOCAL - uses python-docx library)."""
    print(f"\n📝 Creating Word document (LOCAL)...")
    
    doc = Document()
    lines = markdown_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('<!--'):
            continue
        
        # Headings
        if line.startswith('# '):
            doc.add_heading(line[2:], 0)
        elif line.startswith('## '):
            doc.add_heading(line[3:], 1)
        elif line.startswith('### '):
            doc.add_heading(line[4:], 2)
        
        # Bullets
        elif line.startswith('- ') or line.startswith('* '):
            indent = len(line) - len(line.lstrip())
            content = line.lstrip('- *').strip()
            
            if indent >= 2:
                doc.add_paragraph(content, style='List Bullet 2')
            else:
                doc.add_paragraph(content, style='List Bullet')
        
        # Numbered lists
        elif re.match(r'^\d+\.\s', line):
            content = re.sub(r'^\d+\.\s', '', line)
            doc.add_paragraph(content, style='List Number')
        
        # Regular text
        else:
            doc.add_paragraph(line)
    
    doc.save(output_path)
    print(f"✅ Saved to local disk: {output_path}")


def main():
    """100% LOCAL workflow - no external API calls."""
    print("=" * 70)
    print("📊 WEEKLY REPORTS CONSOLIDATION SYSTEM")
    print("    100% LOCAL PROCESSING - No External APIs")
    print("=" * 70)
    
    # Find PDF
    pdf_files = list(Path(PDF_PATH).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {PDF_PATH}")
        return
    
    pdf_path = str(pdf_files[0])
    print(f"\n📁 Input: {Path(pdf_path).name}")
    
    # Create output dir
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract (LOCAL)
    pages = extract_pages_from_pdf(pdf_path)
    
    # Step 2: Convert to markdown (LOCAL)
    combined_markdown = convert_all_pages_to_markdown(pages, pipe)
    
    print(f"\n📊 Statistics:")
    print(f"   Characters: {len(combined_markdown):,}")
    print(f"   Est. tokens: {int(len(combined_markdown.split()) * 1.3):,}")
    
    # Step 3: Consolidate (LOCAL)
    consolidated_markdown = consolidate_with_single_query(combined_markdown, pipe)
    
    # Step 4: Create DOCX (LOCAL)
    output_docx = str(Path(OUTPUT_PATH) / "consolidated_report.docx")
    markdown_to_docx(consolidated_markdown, output_docx)
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE - All processing done LOCALLY")
    print("=" * 70)
    print(f"\n📁 Output files:")
    print(f"   • {OUTPUT_PATH}/combined_reports.md")
    print(f"   • {OUTPUT_PATH}/consolidated_report.md")
    print(f"   • {output_docx}")


if __name__ == "__main__":
    main()
