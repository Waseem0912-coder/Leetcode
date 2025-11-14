#!/usr/bin/env python3
"""
100% LOCAL weekly reports consolidation system.
No external API calls - all processing happens on your hardware.

Usage:
    python consolidate_reports.py --pdf /path/to/reports.pdf --output /path/to/output
    python consolidate_reports.py --pdf reports.pdf  # Uses ./output by default
"""
import argparse
from pathlib import Path
from typing import List, Dict
import pdfplumber
from docx import Document
from transformers import pipeline
import torch
import re
import os
from datetime import datetime

# Verify we're running locally
def print_environment_info():
    """Display system information for local processing verification."""
    print("=" * 70)
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected - will run on CPU (slower)")
    print(f"HuggingFace cache: {os.environ.get('HF_HOME', '~/.cache/huggingface/')}")
    print("=" * 70 + "\n")


def load_model(model_id: str = "openai/gpt-oss-120b"):
    """Load the local LLM model."""
    print("🔧 Loading GPT-OSS-120B model (LOCAL)...")
    print("   Note: First run will download ~240GB from HuggingFace")
    print("   Subsequent runs use cached local copy\n")
    
    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",  # Uses float16/bfloat16 automatically
            device_map="auto",   # Distributes across available GPUs
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully - All inference will be LOCAL\n")
        return pipe
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from PDF in one go (LOCAL - uses pdfplumber library)."""
    print(f"📄 Extracting text from: {pdf_path.name}")
    
    try:
        all_text = []
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"   Found {total_pages} pages")
            
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    all_text.append(f"\n--- Page {i} ---\n")
                    all_text.append(page_text.strip())
                
                if i % 10 == 0 or i == total_pages:
                    print(f"   Progress: {i}/{total_pages} pages...", end='\r')
            
            print()  # New line after progress
                    
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        raise
    
    combined_text = "\n\n".join(all_text)
    print(f"✅ Extracted {total_pages} pages ({len(combined_text):,} characters)\n")
    return combined_text


def chunk_text(text: str, max_chars: int = 80000) -> List[str]:
    """
    Split text into chunks if it's too large.
    Tries to split on page boundaries or paragraphs.
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Split on page markers
    sections = text.split("\n--- Page ")
    
    for section in sections:
        section = ("--- Page " + section) if section != sections[0] else section
        section_size = len(section)
        
        if current_size + section_size > max_chars and current_chunk:
            # Save current chunk and start new one
            chunks.append("\n".join(current_chunk))
            current_chunk = [section]
            current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks


def convert_to_markdown(text: str, pipe, chunk_num: int = 1, total_chunks: int = 1) -> str:
    """
    Convert text to markdown using LOCAL LLM inference.
    NO external API calls - runs entirely on your GPU/CPU.
    """
    
    print(f"🔄 Converting to markdown (chunk {chunk_num}/{total_chunks})...")
    
    # Estimate tokens
    words = len(text.split())
    approx_tokens = int(words * 1.3)
    print(f"   Input: ~{approx_tokens:,} tokens")
    
    messages = [
        {
            "role": "user",
            "content": f"""Convert the following weekly reports to clean, structured markdown format.

REQUIREMENTS:
- Preserve ALL information: numbers, dates, names, metrics, achievements
- Use # for main headings (major topics/weeks)
- Use ## for subheadings (subtopics, projects)
- Use ### for minor headings
- Use bullet points (-) for lists
- Keep chronological order where applicable
- Remove redundant page markers
- Make it scannable and well-organized

TEXT TO CONVERT:
{text}

OUTPUT ONLY THE MARKDOWN - NO EXPLANATIONS OR PREAMBLE."""
        }
    ]
    
    try:
        # LOCAL INFERENCE
        outputs = pipe(
            messages,
            max_new_tokens=4096,
            do_sample=False,
            temperature=0.1,
        )
        
        # Extract the generated text correctly
        generated = outputs[0]["generated_text"]
        
        # The output is a list of messages, get the last assistant message
        if isinstance(generated, list):
            markdown = generated[-1]["content"] if "content" in generated[-1] else str(generated[-1])
        else:
            markdown = str(generated)
        
        print(f"✅ Conversion complete ({len(markdown):,} characters)\n")
        return markdown
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        raise


def consolidate_reports(markdown_text: str, pipe) -> str:
    """
    Final consolidation using LOCAL LLM.
    This is the main processing step - creates a comprehensive summary.
    """
    print("🤖 FINAL CONSOLIDATION (LOCAL inference)...")
    print("=" * 70)
    
    # Calculate token estimate
    words = len(markdown_text.split())
    approx_tokens = int(words * 1.3)
    print(f"Input size: ~{approx_tokens:,} tokens")
    print(f"Model context: ~120,000 tokens")
    
    if approx_tokens > 100000:
        print("⚠️  WARNING: Approaching context limit - consider processing fewer weeks")
    else:
        print("✓ Input fits comfortably in context")
    
    print("\nThis may take several minutes depending on your hardware...\n")
    
    messages = [
        {
            "role": "user",
            "content": f"""You are analyzing 45 weeks of weekly reports. Create a comprehensive, consolidated summary.

CONSOLIDATION REQUIREMENTS:

1. STRUCTURE - Organize by major themes/topics, not by week:
   # Executive Summary
   - High-level overview with key metrics and achievements
   
   # Major Projects & Initiatives
   ## [Project Name]
   - Objectives and outcomes
   - Key milestones achieved
   - Metrics and KPIs
   
   # Key Achievements
   - Significant accomplishments with dates
   - Impact and results
   
   # Challenges & Solutions
   - Major obstacles encountered
   - How they were resolved
   
   # Metrics & Data
   - All quantitative results
   - Trends over time
   
   # Action Items & Next Steps
   - Outstanding tasks
   - Future priorities

2. CONTENT RULES:
   - Preserve ALL specific data: numbers, percentages, dates, names
   - Combine related information from different weeks
   - Remove redundancy but keep important details
   - Highlight trends and patterns across weeks
   - Include both successes AND challenges
   - Be specific and concrete

3. SYNTHESIS:
   - Don't just list items - synthesize patterns
   - Show progression and evolution over time
   - Connect related initiatives
   - Identify what worked and what didn't

WEEKLY REPORTS TO CONSOLIDATE:
{markdown_text}

OUTPUT: Comprehensive consolidated markdown report following the structure above."""
        }
    ]
    
    try:
        # LOCAL INFERENCE - the main consolidation
        outputs = pipe(
            messages,
            max_new_tokens=8192,
            do_sample=False,
            temperature=0.1,
        )
        
        # Extract the generated text correctly
        generated = outputs[0]["generated_text"]
        
        if isinstance(generated, list):
            consolidated = generated[-1]["content"] if "content" in generated[-1] else str(generated[-1])
        else:
            consolidated = str(generated)
        
        print("✅ CONSOLIDATION COMPLETE\n")
        return consolidated
        
    except Exception as e:
        print(f"❌ Error during consolidation: {e}")
        raise


def markdown_to_docx(markdown_text: str, output_path: Path):
    """Convert markdown to DOCX (LOCAL - uses python-docx library)."""
    print(f"📝 Creating Word document...")
    
    doc = Document()
    
    # Add title and metadata
    title = doc.add_heading('Consolidated Weekly Reports', 0)
    doc.add_paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    doc.add_paragraph()  # Blank line
    
    lines = markdown_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and HTML comments
        if not line or line.startswith('<!--'):
            continue
        
        # Headings
        if line.startswith('# '):
            doc.add_heading(line[2:], 1)
        elif line.startswith('## '):
            doc.add_heading(line[3:], 2)
        elif line.startswith('### '):
            doc.add_heading(line[4:], 3)
        elif line.startswith('#### '):
            doc.add_heading(line[5:], 4)
        
        # Bullets
        elif line.startswith('- ') or line.startswith('* '):
            # Calculate indent level
            original_line = line
            indent = 0
            while line.startswith('  - ') or line.startswith('  * '):
                indent += 1
                line = line[2:]
            
            content = line.lstrip('- *').strip()
            
            if indent >= 2:
                doc.add_paragraph(content, style='List Bullet 2')
            elif indent == 1:
                try:
                    doc.add_paragraph(content, style='List Bullet 2')
                except:
                    doc.add_paragraph(content, style='List Bullet')
            else:
                doc.add_paragraph(content, style='List Bullet')
        
        # Numbered lists
        elif re.match(r'^\d+\.\s', line):
            content = re.sub(r'^\d+\.\s', '', line)
            doc.add_paragraph(content, style='List Number')
        
        # Bold text (preserve formatting)
        elif '**' in line:
            p = doc.add_paragraph()
            parts = line.split('**')
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    p.add_run(part)
                else:
                    p.add_run(part).bold = True
        
        # Regular text
        else:
            doc.add_paragraph(line)
    
    doc.save(str(output_path))
    print(f"✅ Word document saved: {output_path}\n")


def main():
    """100% LOCAL workflow - no external API calls."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Consolidate weekly reports using local LLM - 100% offline processing'
    )
    parser.add_argument(
        '--pdf',
        type=str,
        required=True,
        help='Path to the PDF file containing weekly reports'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for generated files (default: ./output)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='openai/gpt-oss-120b',
        help='HuggingFace model ID (default: openai/gpt-oss-120b)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output)
    
    # Validate PDF exists
    if not pdf_path.exists():
        print(f"❌ ERROR: PDF file not found: {pdf_path}")
        return 1
    
    if not pdf_path.is_file():
        print(f"❌ ERROR: Path is not a file: {pdf_path}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}\n")
    
    # Display environment info
    print_environment_info()
    
    print("=" * 70)
    print("📊 WEEKLY REPORTS CONSOLIDATION SYSTEM")
    print("    100% LOCAL PROCESSING - No External APIs")
    print("=" * 70)
    print(f"\n📄 Input: {pdf_path}")
    print(f"📁 Output: {output_dir}\n")
    
    # Load model
    pipe = load_model(args.model)
    
    # Step 1: Extract text from PDF (LOCAL)
    full_text = extract_text_from_pdf(pdf_path)
    
    # Save raw extracted text
    raw_text_path = output_dir / "01_raw_extracted_text.txt"
    raw_text_path.write_text(full_text, encoding='utf-8')
    print(f"💾 Saved raw text: {raw_text_path}\n")
    
    # Step 2: Convert to markdown (LOCAL)
    # Check if we need to chunk
    chunks = chunk_text(full_text, max_chars=80000)
    
    if len(chunks) > 1:
        print(f"📦 Text split into {len(chunks)} chunks for processing\n")
        markdown_parts = []
        for i, chunk in enumerate(chunks, 1):
            md = convert_to_markdown(chunk, pipe, chunk_num=i, total_chunks=len(chunks))
            markdown_parts.append(md)
        combined_markdown = "\n\n".join(markdown_parts)
    else:
        combined_markdown = convert_to_markdown(full_text, pipe)
    
    # Save combined markdown
    markdown_path = output_dir / "02_combined_reports.md"
    markdown_path.write_text(combined_markdown, encoding='utf-8')
    print(f"💾 Saved markdown: {markdown_path}\n")
    
    print(f"📊 STATISTICS:")
    print(f"   Characters: {len(combined_markdown):,}")
    print(f"   Words: {len(combined_markdown.split()):,}")
    print(f"   Est. tokens: {int(len(combined_markdown.split()) * 1.3):,}\n")
    
    # Step 3: Consolidate (LOCAL)
    consolidated_markdown = consolidate_reports(combined_markdown, pipe)
    
    # Save consolidated markdown
    consolidated_md_path = output_dir / "03_consolidated_report.md"
    consolidated_md_path.write_text(consolidated_markdown, encoding='utf-8')
    print(f"💾 Saved consolidated markdown: {consolidated_md_path}\n")
    
    # Step 4: Create DOCX (LOCAL)
    output_docx = output_dir / "04_consolidated_report.docx"
    markdown_to_docx(consolidated_markdown, output_docx)
    
    # Final summary
    print("=" * 70)
    print("✅ PROCESSING COMPLETE - All done LOCALLY on your hardware")
    print("=" * 70)
    print(f"\n📁 OUTPUT FILES in {output_dir}:")
    print(f"   1. {raw_text_path.name} - Raw extracted text")
    print(f"   2. {markdown_path.name} - Markdown version")
    print(f"   3. {consolidated_md_path.name} - Consolidated markdown")
    print(f"   4. {output_docx.name} - Final Word document")
    print("\n🎉 Success! Your consolidated report is ready.\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
