#!/usr/bin/env python3
"""
STREAMLINED: Extract 45 weeks of reports and combine into ONE organized document.
No unnecessary conversions - just extract, organize, and output.
"""
import argparse
from pathlib import Path
import pdfplumber
from docx import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from datetime import datetime

def print_environment_info():
    """Display GPU configuration."""
    print("=" * 70)
    print("🔍 GPU CONFIGURATION")
    print("=" * 70)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\nDetected {num_gpus} GPUs:")
        total_memory = 0
        for i in range(num_gpus):
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            total_memory += gpu_mem
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {gpu_mem:.1f} GB")
        print(f"\nTotal VRAM: {total_memory:.1f} GB")
    else:
        print("\n⚠️  No GPU detected - will run on CPU")
    print("=" * 70 + "\n")


def load_model(model_id: str = "openai/gpt-oss-120b"):
    """Load model distributed across all GPUs."""
    print(f"🔧 Loading {model_id}...")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")
    
    # Configure memory per GPU (leave headroom for activations/KV cache)
    max_memory_per_gpu = {}
    for i in range(num_gpus):
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        # For H200 (141GB), reserve 70GB for model, 70GB for operations
        usable_mem = max(total_mem * 0.5, 60)
        max_memory_per_gpu[i] = f"{int(usable_mem)}GiB"
    
    print(f"\n💾 Memory allocation per GPU: {list(max_memory_per_gpu.values())[0]}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="balanced",  # BALANCED distribution
        max_memory=max_memory_per_gpu,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Verify distribution
    if hasattr(model, 'hf_device_map'):
        print("\n📍 Model Distribution:")
        device_counts = {}
        for name, device in model.hf_device_map.items():
            device_str = f"GPU {device}" if isinstance(device, int) else str(device)
            device_counts[device_str] = device_counts.get(device_str, 0) + 1
        
        for device, count in sorted(device_counts.items()):
            print(f"  {device}: {count} layers")
    
    print("\n✅ Model loaded and distributed across all GPUs\n")
    return model, tokenizer


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from PDF - simple and direct."""
    print(f"📄 Extracting text from: {pdf_path.name}")
    
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Found {total_pages} pages")
        
        for i, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                # Keep page markers for reference
                all_text.append(f"\n{'='*70}\nPAGE {i}\n{'='*70}\n")
                all_text.append(page_text.strip())
            
            if i % 10 == 0 or i == total_pages:
                print(f"   Progress: {i}/{total_pages} pages...", end='\r')
        
        print()
    
    combined_text = "\n\n".join(all_text)
    print(f"✅ Extracted {total_pages} pages ({len(combined_text):,} characters)\n")
    return combined_text


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 16384) -> str:
    """
    Generate text using ALL GPUs properly.
    No hardcoded cuda:0 - uses model's device map.
    """
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=120000)
    
    # Move to the FIRST device in the model's device map (not hardcoded cuda:0)
    # This finds where the model's first layer actually lives
    first_device = None
    if hasattr(model, 'hf_device_map'):
        # Get the device of the first layer
        for name, device in model.hf_device_map.items():
            first_device = device
            break
    
    if first_device is None:
        first_device = next(model.parameters()).device
    
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    
    print(f"   Input tokens: {inputs['input_ids'].shape[1]:,}")
    print(f"   Max output tokens: {max_new_tokens:,}")
    print(f"   First layer device: {first_device}")
    print(f"   Generating across all GPUs...")
    
    # Generate with proper settings
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # MUCH LARGER - can output ~12k tokens
            do_sample=False,  # Deterministic
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only new tokens
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return generated_text


def consolidate_all_reports(text: str, model, tokenizer) -> str:
    """
    ONE consolidated document - organized but keeping all detail.
    This is the main processing step.
    """
    print("🤖 CONSOLIDATING ALL 45 WEEKS...")
    print("=" * 70)
    
    words = len(text.split())
    approx_tokens = int(words * 1.3)
    print(f"Input: ~{approx_tokens:,} tokens")
    print(f"Context window: ~120,000 tokens\n")
    
    if approx_tokens > 100000:
        print("⚠️  Large input - this will take time\n")
    
    # THE PROMPT - tells it to organize but keep everything
    prompt = f"""
You are analyzing a long text containing ~45 weeks of weekly reports.  
Each week contains several main bullets (topics/projects) and sub-bullets (details, updates, tasks).  
Some topics appear in multiple weeks with different updates, some appear only once, and some weeks reset or replace previous information.

GOAL  
Reconstruct the entire set of weekly reports into a clean, structured markdown document that:
- Preserves the original week-by-week structure
- Preserves the original topic names EXACTLY as they appeared (no normalization or merging)
- Preserves all details under each week and under the correct topic
- Makes the final document readable, scannable, and consistently formatted

INTERNAL REASONING STEPS (DO NOT OUTPUT THESE STEPS)  
1. Detect week boundaries (Week 1 → Week 45 or similar markers).  
2. Inside each week:
   - Identify which bullets are main topics (first-level bullets or bolded headers).
   - Identify which bullets are subpoints or child items.
3. Do NOT normalize, merge, rename, or cluster topics.
   - If Week 3 says “Offline Monitoring” and Week 7 says “Monitoring Improvements,” treat them as two separate topics even if they seem similar.
4. Do NOT build a cross-week topic timeline.
   - Each week is self-contained.
   - Simply reconstruct Week 1’s content, then Week 2’s content, etc.
5. Preserve:
   - All numbers, dates, metrics, names, file paths, schemas, decisions, blockers, achievements.
   - All bullet points and nested bullets.
   - The order bullets appeared in the original text.
6. Clean up:
   - Remove PDF artifacts, page numbers, broken lines, nonsense breaks.
   - Convert inconsistent indentation into consistent markdown hierarchy.

OUTPUT FORMAT  
Use the following exact structure:

- `# Week X` for each week  
- Under each week, use `## <topic name exactly as written>`  
- Under each topic, use bullet points (`-`) and nested bullets as needed  
- Keep chronological order (Week 1 → Week 45)  
- Keep the original content attached to its week only  
- Do NOT merge or reorganize topics across weeks  
- Do NOT infer missing continuity or relationships

ADDITIONAL REQUIREMENTS  
- Output ONLY the final markdown.  
- NO explanations, NO preamble, NO reasoning.  
- No hallucination: use only the information in the provided text.

TEXT TO CONVERT:
{text}
"""
    
    print("Processing (this may take 5-10 minutes)...")
    consolidated = generate_text(model, tokenizer, prompt, max_new_tokens=16384)
    
    print(f"\n✅ Generated {len(consolidated):,} characters\n")
    return consolidated


def markdown_to_docx(markdown_text: str, output_path: Path):
    """Convert markdown to professional Word document."""
    print(f"📝 Creating Word document...")
    
    doc = Document()
    
    # Add title page
    title = doc.add_heading('Consolidated Weekly Reports', 0)
    doc.add_paragraph(f'45 Weeks Consolidated')
    doc.add_paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    doc.add_page_break()
    
    lines = markdown_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:
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
        
        # Bullet points
        elif line.startswith('- ') or line.startswith('* '):
            content = line.lstrip('- *').strip()
            # Handle bold text
            if '**' in content:
                p = doc.add_paragraph(style='List Bullet')
                parts = content.split('**')
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        p.add_run(part)
                    else:
                        p.add_run(part).bold = True
            else:
                doc.add_paragraph(content, style='List Bullet')
        
        # Numbered lists
        elif re.match(r'^\d+\.\s', line):
            content = re.sub(r'^\d+\.\s', '', line)
            doc.add_paragraph(content, style='List Number')
        
        # Regular paragraphs
        else:
            if '**' in line:
                p = doc.add_paragraph()
                parts = line.split('**')
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        p.add_run(part)
                    else:
                        p.add_run(part).bold = True
            else:
                doc.add_paragraph(line)
    
    doc.save(str(output_path))
    print(f"✅ Saved: {output_path}\n")


def main():
    """Streamlined workflow: Extract → Consolidate → Output"""
    
    parser = argparse.ArgumentParser(
        description='Consolidate 45 weeks of reports into one organized document'
    )
    parser.add_argument('--pdf', type=str, required=True, help='Path to PDF with all reports')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--model', type=str, default='openai/gpt-oss-120b', help='Model ID')
    parser.add_argument('--max-tokens', type=int, default=16384, help='Max output tokens (default: 16384)')
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output)
    
    if not pdf_path.exists():
        print(f"❌ ERROR: PDF not found: {pdf_path}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_environment_info()
    
    print("=" * 70)
    print("📊 WEEKLY REPORTS CONSOLIDATION - STREAMLINED")
    print("=" * 70)
    print(f"\n📄 Input: {pdf_path}")
    print(f"📁 Output: {output_dir}\n")
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Extract PDF (raw text)
    raw_text = extract_text_from_pdf(pdf_path)
    
    # Save raw extraction
    raw_path = output_dir / "01_raw_extracted_text.txt"
    raw_path.write_text(raw_text, encoding='utf-8')
    print(f"💾 Saved raw text: {raw_path}\n")
    
    # ONE consolidation step - this is where the magic happens
    consolidated_md = consolidate_all_reports(raw_text, model, tokenizer)
    
    # Save markdown version
    md_path = output_dir / "02_consolidated_report.md"
    md_path.write_text(consolidated_md, encoding='utf-8')
    print(f"💾 Saved markdown: {md_path}\n")
    
    # Create Word document
    docx_path = output_dir / "03_consolidated_report.docx"
    markdown_to_docx(consolidated_md, docx_path)
    
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print(f"\n📁 Outputs in {output_dir}:")
    print(f"   1. {raw_path.name} - Raw extracted text")
    print(f"   2. {md_path.name} - Consolidated markdown")
    print(f"   3. {docx_path.name} - Professional Word document")
    print(f"\n🎉 Your 45 weeks are now organized in one document!\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
