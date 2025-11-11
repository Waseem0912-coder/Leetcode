Here's the updated script with the original prompt structure, proper markdown parsing, and weekly report consolidation:

```python
"""
RAG-based PDF Analysis System with Quantization and DOCX Output
This script processes weekly PDF reports, creates embeddings using quantization for speed,
and generates consolidated reports in DOCX format.
"""

import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Tuple
import PyPDF2
import numpy as np
from pathlib import Path
import json
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime
import re

# ============================================================================
# MODEL DOWNLOAD AND CACHE MANAGEMENT
# ============================================================================

def ensure_model_downloaded(model_name: str, model_type: str = "embedding"):
    """
    Ensure model is downloaded locally before loading.
    
    Args:
        model_name: HuggingFace model identifier
        model_type: Type of model ("embedding" or "inference")
    """
    from huggingface_hub import snapshot_download
    from transformers.utils import TRANSFORMERS_CACHE
    
    print(f"\n{'='*80}")
    print(f"Checking {model_type} model: {model_name}")
    print(f"{'='*80}")
    
    # Get cache directory
    cache_dir = os.environ.get('HF_HOME', TRANSFORMERS_CACHE)
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Try to download/verify the model
        print(f"Downloading/verifying model files...")
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        print(f"✓ Model available at: {local_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False


# ============================================================================
# EMBEDDING MODEL SETUP (Qwen3-Embedding-0.6B) with Quantization
# ============================================================================

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last token from the hidden states."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format instruction for embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'


class EmbeddingModel:
    """Handles text embedding generation using Qwen3-Embedding-0.6B with quantization."""
    
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B', use_quantization=True):
        self.model_name = model_name
        
        print(f"\n{'='*80}")
        print("INITIALIZING EMBEDDING MODEL (WITH QUANTIZATION)")
        print(f"{'='*80}")
        
        if not ensure_model_downloaded(model_name, "embedding"):
            raise RuntimeError(f"Failed to download embedding model: {model_name}")
        
        print(f"\nLoading embedding model into memory...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side='left',
            local_files_only=False,
            resume_download=True
        )
        
        # Load model with FP16 for faster inference
        if use_quantization and torch.cuda.is_available():
            print("Using FP16 precision for faster inference on GPU...")
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=False,
                resume_download=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            print("Loading model in default precision...")
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=False,
                resume_download=True
            )
        
        self.max_length = 8192
        self.device = next(self.model.parameters()).device
        print(f"✓ Embedding model loaded successfully on device: {self.device}")
    
    def create_embeddings(self, texts: List[str], task_description: str = None, batch_size: int = 8) -> Tensor:
        """
        Create embeddings for a list of texts with batching for efficiency.
        
        Args:
            texts: List of text strings to embed
            task_description: Optional task description for queries
            batch_size: Number of texts to process at once
        
        Returns:
            Normalized embeddings tensor
        """
        # Add instruction prefix if task description provided
        if task_description:
            input_texts = [get_detailed_instruct(task_description, text) for text in texts]
        else:
            input_texts = texts
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]
            
            # Show progress
            if i % (batch_size * 5) == 0:
                print(f"  Processing embeddings: {i}/{len(input_texts)} texts...")
            
            # Tokenize
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict = batch_dict.to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Normalize
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        return all_embeddings


# ============================================================================
# INFERENCE MODEL SETUP (Qwen3-1.7B) with 4-bit Quantization
# ============================================================================

class InferenceModel:
    """Handles text generation using Qwen3-1.7B with 4-bit quantization."""
    
    def __init__(self, model_name="Qwen/Qwen3-1.7B", use_quantization=True):
        self.model_name = model_name
        
        print(f"\n{'='*80}")
        print("INITIALIZING INFERENCE MODEL (WITH 4-BIT QUANTIZATION)")
        print(f"{'='*80}")
        
        if not ensure_model_downloaded(model_name, "inference"):
            raise RuntimeError(f"Failed to download inference model: {model_name}")
        
        print(f"\nLoading inference model into memory...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=False,
            resume_download=True
        )
        
        # Load with 4-bit quantization if available
        if use_quantization and torch.cuda.is_available():
            print("Using 4-bit quantization (BitsAndBytes) for inference...")
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    local_files_only=False,
                    resume_download=True
                )
            except Exception as e:
                print(f"⚠ Quantization failed ({e}), loading in FP16...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    local_files_only=False,
                    resume_download=True
                )
        else:
            print("Loading model in default precision...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=False,
                resume_download=True
            )
        
        print(f"✓ Inference model loaded successfully")
    
    def generate_response(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.1) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate with low temperature to reduce hallucinations
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
        except ValueError:
            index = 0
        
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content


# ============================================================================
# PDF PROCESSING
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


# ============================================================================
# DOCX REPORT GENERATION WITH MARKDOWN PARSING
# ============================================================================

class DOCXReportGenerator:
    """Generate professional DOCX reports with proper markdown parsing."""
    
    def __init__(self, filename: str = "rag_analysis_report.docx"):
        self.filename = filename
        self.document = Document()
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup document styles."""
        # Set default font
        style = self.document.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)
    
    def add_title(self, title: str):
        """Add title to document."""
        heading = self.document.add_heading(title, level=0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = heading.runs[0]
        run.font.size = Pt(24)
        run.font.color.rgb = RGBColor(0, 51, 102)
    
    def add_metadata(self, metadata: Dict[str, any]):
        """Add metadata section."""
        self.document.add_heading('Report Metadata', level=1)
        
        table = self.document.add_table(rows=len(metadata), cols=2)
        table.style = 'Light Grid Accent 1'
        
        for i, (key, value) in enumerate(metadata.items()):
            row = table.rows[i]
            row.cells[0].text = str(key)
            row.cells[1].text = str(value)
            
            # Bold the key
            row.cells[0].paragraphs[0].runs[0].font.bold = True
        
        self.document.add_paragraph()
    
    def parse_and_add_markdown(self, content: str):
        """
        Parse markdown content and add to document with proper formatting.
        
        Args:
            content: Markdown formatted text
        """
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Handle headers
            if line.startswith('# '):
                # H1
                text = line.lstrip('#').strip()
                self.document.add_heading(text, level=1)
            elif line.startswith('## '):
                # H2
                text = line.lstrip('#').strip()
                self.document.add_heading(text, level=2)
            elif line.startswith('### '):
                # H3
                text = line.lstrip('#').strip()
                self.document.add_heading(text, level=3)
            elif line.startswith('#### '):
                # H4
                text = line.lstrip('#').strip()
                self.document.add_heading(text, level=4)
            
            # Handle horizontal rules
            elif line.startswith('---') or line.startswith('***'):
                self.document.add_paragraph('_' * 80)
            
            # Handle bullet points
            elif line.startswith('- ') or line.startswith('* '):
                text = line.lstrip('-*').strip()
                text = self._process_inline_formatting(text)
                p = self.document.add_paragraph(style='List Bullet')
                self._add_formatted_text(p, text)
            
            # Handle numbered lists
            elif re.match(r'^\d+\.\s', line):
                text = re.sub(r'^\d+\.\s', '', line)
                text = self._process_inline_formatting(text)
                p = self.document.add_paragraph(style='List Number')
                self._add_formatted_text(p, text)
            
            # Handle blockquotes
            elif line.startswith('> '):
                text = line.lstrip('>').strip()
                text = self._process_inline_formatting(text)
                p = self.document.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.5)
                self._add_formatted_text(p, text)
                run = p.runs[0]
                run.italic = True
            
            # Regular paragraph
            else:
                text = self._process_inline_formatting(line)
                p = self.document.add_paragraph()
                self._add_formatted_text(p, text)
            
            i += 1
    
    def _process_inline_formatting(self, text: str) -> str:
        """Process inline markdown formatting markers."""
        # We'll handle this in _add_formatted_text
        return text
    
    def _add_formatted_text(self, paragraph, text: str):
        """
        Add text to paragraph with inline formatting (bold, italic, code).
        
        Args:
            paragraph: Document paragraph object
            text: Text with markdown formatting
        """
        # Pattern to match bold, italic, and code
        pattern = r'(\*\*\*.*?\*\*\*|\*\*.*?\*\*|\*.*?\*|`.*?`)'
        parts = re.split(pattern, text)
        
        for part in parts:
            if not part:
                continue
            
            # Bold and italic
            if part.startswith('***') and part.endswith('***'):
                run = paragraph.add_run(part[3:-3])
                run.bold = True
                run.italic = True
            # Bold
            elif part.startswith('**') and part.endswith('**'):
                run = paragraph.add_run(part[2:-2])
                run.bold = True
            # Italic
            elif part.startswith('*') and part.endswith('*'):
                run = paragraph.add_run(part[1:-1])
                run.italic = True
            # Code
            elif part.startswith('`') and part.endswith('`'):
                run = paragraph.add_run(part[1:-1])
                run.font.name = 'Courier New'
                run.font.size = Pt(10)
            # Regular text
            else:
                paragraph.add_run(part)
    
    def add_page_break(self):
        """Add a page break."""
        self.document.add_page_break()
    
    def save(self):
        """Save the document."""
        self.document.save(self.filename)
        print(f"\n✓ Report saved to: {self.filename}")


# ============================================================================
# RAG SYSTEM
# ============================================================================

class RAGSystem:
    """RAG system for PDF analysis and report generation."""
    
    def __init__(self, pdf_folder: str = "pdf", use_quantization: bool = True):
        self.pdf_folder = pdf_folder
        self.use_quantization = use_quantization
        self.embedding_model = EmbeddingModel(use_quantization=use_quantization)
        self.inference_model = InferenceModel(use_quantization=use_quantization)
        self.documents = []
        self.embeddings = None
        self.pdf_metadata = []
    
    def load_pdfs(self):
        """Load and process all PDFs in the target folder."""
        pdf_path = Path(self.pdf_folder)
        
        if not pdf_path.exists():
            print(f"\n{'='*80}")
            print(f"Creating folder: {self.pdf_folder}")
            print(f"{'='*80}")
            pdf_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Folder created: {pdf_path.absolute()}")
            print(f"\n⚠ Please add PDF files to the '{self.pdf_folder}' folder and run again.")
            return
        
        pdf_files = sorted(list(pdf_path.glob("*.pdf")))  # Sort for consistent ordering
        
        if not pdf_files:
            print(f"\n⚠ No PDF files found in '{self.pdf_folder}' folder.")
            print(f"   Location: {pdf_path.absolute()}")
            return
        
        print(f"\n{'='*80}")
        print(f"LOADING WEEKLY REPORT PDFs")
        print(f"{'='*80}")
        print(f"Found {len(pdf_files)} weekly report(s). Processing...\n")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            text = extract_text_from_pdf(str(pdf_file))
            
            if text:
                # Chunk the text
                chunks = chunk_text(text, chunk_size=1500, overlap=300)
                
                for i, chunk in enumerate(chunks):
                    self.documents.append(chunk)
                    self.pdf_metadata.append({
                        'filename': pdf_file.name,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    })
                
                print(f"  ✓ Extracted {len(chunks)} chunks")
        
        print(f"\n✓ Loaded {len(self.documents)} text chunks from {len(pdf_files)} weekly report(s).")
    
    def create_embeddings(self, batch_size: int = 16):
        """Create embeddings for all document chunks with batching."""
        if not self.documents:
            print("\n⚠ No documents to embed. Load PDFs first.")
            return
        
        print(f"\n{'='*80}")
        print("CREATING EMBEDDINGS")
        print(f"{'='*80}")
        print(f"Creating embeddings for {len(self.documents)} document chunks...")
        print(f"Using batch size: {batch_size}")
        
        self.embeddings = self.embedding_model.create_embeddings(
            self.documents, 
            batch_size=batch_size
        )
        print(f"\n✓ Created embeddings with shape: {self.embeddings.shape}")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of (chunk_text, similarity_score, metadata) tuples
        """
        if self.embeddings is None:
            print("⚠ No embeddings available. Create embeddings first.")
            return []
        
        # Create query embedding
        task = "Given a document analysis task, retrieve relevant sections that contain the information"
        query_embedding = self.embedding_model.create_embeddings([query], task_description=task)
        
        # Calculate similarities
        similarities = (query_embedding @ self.embeddings.T).squeeze(0)
        
        # Get top-k indices
        top_k_indices = torch.topk(similarities, min(top_k, len(similarities))).indices.tolist()
        
        # Return results
        results = []
        for idx in top_k_indices:
            results.append((
                self.documents[idx],
                similarities[idx].item(),
                self.pdf_metadata[idx]
            ))
        
        return results
    
    def generate_comprehensive_report(self, top_k_per_topic: int = 15):
        """
        Generate a comprehensive consolidated report from all weekly reports.
        
        Args:
            top_k_per_topic: Number of chunks to retrieve per analysis
        """
        if not self.documents:
            print("\n⚠ No documents loaded. Please add PDFs and try again.")
            return
        
        print(f"\n{'='*80}")
        print("GENERATING CONSOLIDATED WEEKLY REPORT")
        print(f"{'='*80}")
        
        # Initialize DOCX generator
        docx_gen = DOCXReportGenerator("consolidated_weekly_report.docx")
        docx_gen.add_title("Consolidated Weekly Report Analysis")
        
        # Add metadata
        unique_pdfs = sorted(list(set([m['filename'] for m in self.pdf_metadata])))
        metadata = {
            "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total Weekly Reports": len(unique_pdfs),
            "Report Files": ", ".join(unique_pdfs),
            "Total Sections Analyzed": len(self.documents),
            "Analysis Method": "RAG with Semantic Consolidation"
        }
        docx_gen.add_metadata(metadata)
        docx_gen.add_page_break()
        
        # Retrieve comprehensive chunks for analysis
        print("\nAnalyzing all weekly reports for unique information...")
        
        analysis_chunks = self.retrieve_relevant_chunks(
            "all topics events activities findings updates progress issues accomplishments", 
            top_k=min(30, len(self.documents))
        )
        
        full_context = "\n\n".join([
            f"[Week Report: {meta['filename']}, Section {meta['chunk_id']+1}]\n{chunk}" 
            for chunk, _, meta in analysis_chunks
        ])
        
        # Enhanced prompt for weekly report consolidation
        consolidation_prompt = f"""You are analyzing multiple weekly reports to create a consolidated summary. Each PDF represents a different week's report.

TASK: Extract and organize all UNIQUE events, activities, findings, and updates across all weeks.

CRITICAL RULES:
1. ONLY include information explicitly stated in the weekly reports
2. Each unique point should be mentioned ONLY ONCE
3. Group similar topics together
4. Cite the source week for each point (e.g., "Week 1.pdf", "Week 2.pdf")
5. If the same event appears in multiple weeks, note it only once with all relevant week references
6. DO NOT add information not present in the reports
7. DO NOT make assumptions or inferences

WEEKLY REPORTS CONTENT:
{full_context}

Create a comprehensive consolidated report in the following format:

# COMPREHENSIVE ANALYSIS REPORT

## Topic/Section 1: [Name]
- Unique finding/event 1 (Source: Week_X.pdf)
- Unique finding/event 2 (Source: Week_Y.pdf)
- Recurring item mentioned in multiple weeks (Sources: Week_X.pdf, Week_Y.pdf, Week_Z.pdf)

## Topic/Section 2: [Name]
- Unique finding/event 1 (Source: Week_X.pdf)
- Unique finding/event 2 (Source: Week_Y.pdf)

[Continue for all identified topics]

## Cross-Week Patterns
[Note any patterns or themes that appear across multiple weeks]

IMPORTANT: 
- Every bullet point must cite its source week(s)
- Avoid repetition - each unique piece of information appears only once
- Group related information under appropriate topic headings
- Use clear, descriptive topic headings based on the actual content"""
        
        print("  Generating consolidated analysis...")
        final_report = self.inference_model.generate_response(
            consolidation_prompt, 
            max_tokens=8192, 
            temperature=0.1
        )
        
        print(f"\n{'='*80}")
        print("CONSOLIDATED REPORT")
        print(f"{'='*80}\n")
        print(final_report)
        
        # Parse and add the markdown content to DOCX
        docx_gen.parse_and_add_markdown(final_report)
        
        # Add disclaimer
        docx_gen.add_page_break()
        disclaimer = """This consolidated report was generated using AI-powered analysis of multiple weekly PDF reports. 
The analysis extracts and organizes unique information from each week, eliminating redundancies and grouping related content.

All findings and statements are extracted directly from the source weekly reports and cited accordingly. 
Each item includes references to the specific week(s) where the information was found.

For critical decisions or detailed information, please refer to the original weekly report documents."""
        
        docx_gen.document.add_heading('Disclaimer', level=1)
        for para in disclaimer.split('\n'):
            if para.strip():
                docx_gen.document.add_paragraph(para.strip())
        
        # Save DOCX
        docx_gen.save()
        
        # Also save text version for reference
        with open("consolidated_weekly_report.txt", 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CONSOLIDATED WEEKLY REPORT ANALYSIS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {metadata['Generated']}\n")
            f.write(f"Analyzed {len(unique_pdfs)} weekly report(s)\n")
            f.write(f"Weekly reports: {', '.join(unique_pdfs)}\n")
            f.write(f"Total sections processed: {len(self.documents)}\n\n")
            f.write("="*80 + "\n\n")
            f.write(final_report)
            f.write("\n\n" + "="*80 + "\n")
            f.write("DISCLAIMER\n")
            f.write("="*80 + "\n\n")
            f.write(disclaimer)
        
        print(f"\n✓ Text version also saved to: consolidated_weekly_report.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print("WEEKLY REPORT CONSOLIDATION SYSTEM")
    print("RAG-Based Analysis with Quantization")
    print(f"{'='*80}\n")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ No GPU detected. Running on CPU (will be slower)")
    
    # Initialize RAG system with quantization
    try:
        rag_system = RAGSystem(pdf_folder="pdf", use_quantization=True)
    except Exception as e:
        print(f"\n✗ Error initializing RAG system: {e}")
        print("\nPlease check your internet connection and try again.")
        return
    
    # Load weekly report PDFs
    rag_system.load_pdfs()
    
    if not rag_system.documents:
        print(f"\n{'='*80}")
        print("NO WEEKLY REPORTS TO PROCESS")
        print(f"{'='*80}")
        print("\nExiting. Please add weekly report PDF files and run again.")
        return
    
    # Create embeddings with larger batch size for speed
    rag_system.create_embeddings(batch_size=16)
    
    # Generate consolidated report
    rag_system.generate_comprehensive_report(top_k_per_topic=15)
    
    print(f"\n{'='*80}")
    print("CONSOLIDATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
```

**Key Changes:**

1. **Proper Markdown Parsing**:
   - Added `parse_and_add_markdown()` method that properly handles:
     - Headers (# ## ### ####)
     - Bullet points (- *)
     - Numbered lists (1. 2. 3.)
     - Bold (\*\*text\*\*)
     - Italic (\*text\*)
     - Bold+Italic (\*\*\*text\*\*\*)
     - Code blocks (\`code\`)
     - Horizontal rules (--- ***)
     - Blockquotes (>)
   - No raw markdown symbols in final DOCX

2. **Weekly Report Focus**:
   - Prompt specifically designed for weekly reports
   - Extracts unique points across all weeks
   - Eliminates redundancies
   - Groups by topics/sections
   - Cites source week for each item
   - Notes recurring items across multiple weeks

3. **DOCX Structure**:
   - Report Metadata (kept)
   - Consolidated Report (properly formatted, no markdown symbols)
   - Disclaimer (kept)
   - Only the consolidated analysis in the document

4. **Original Prompt Style**:
   - Topic/Section organization
   - Bullet point format with sources
   - Cross-week patterns section
   - Clear headings and structure

The DOCX output will now be properly formatted with real headings, bold text, bullet points, etc., instead of showing markdown symbols!
