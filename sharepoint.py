import os
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm

# LangChain imports with compatibility handling
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Use the recommended import path for text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("Using langchain_text_splitters (newer version)")

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFacePipeline


# Transformers and quantization imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    pipeline,
    BitsAndBytesConfig
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
    """
    Custom embedding class for Qwen3-Embedding-0.6B model.
    Implements proper pooling and instruction formatting for RAG tasks.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = None):
        """Initialize the custom Qwen embeddings."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading embedding model: {model_name} on {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        self.model.eval()

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool the last token from the sequence for each batch item.

        Args:
            last_hidden_states: Model output hidden states
            attention_mask: Attention mask for the input

        Returns:
            Pooled embeddings
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        """
        Format the query with task-specific instructions.

        Args:
            task_description: Description of the retrieval task
            query: The search query

        Returns:
            Formatted instruction string
        """
        return f'Instruct: {task_description}\nQuery: {query}'

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of document strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []
        batch_size = 8  # Process in batches for efficiency

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Apply pooling
            batch_embeddings = self.last_token_pool(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )

            # Normalize embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            # Convert to list and extend results
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query string to embed

        Returns:
            Embedding vector
        """
        # Format query with instruction
        task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        formatted_query = self.get_detailed_instruct(task_description, text)

        # Tokenize
        inputs = self.tokenizer(
            formatted_query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply pooling
        embedding = self.last_token_pool(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )

        # Normalize
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding[0].cpu().numpy().tolist()


class RAGPipeline:
    """Main RAG pipeline for processing PDFs and generating reports."""

    def __init__(self, source_dir: str = "./source_pdfs", persist_dir: str = "./chroma_db"):
        """
        Initialize the RAG pipeline.

        Args:
            source_dir: Directory containing PDF files
            persist_dir: Directory for persisting vector store
        """
        self.source_dir = source_dir
        self.persist_dir = persist_dir
        self.vector_store = None
        self.llm = None
        self.embeddings = None

    def setup_llm(self):
        """Set up the Qwen3-1.7B language model with 4-bit quantization."""
        logger.info("Setting up Qwen3-1.7B with 4-bit quantization...")

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        model_name = "Qwen/Qwen3-1.7B"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )

        # Create text generation pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

        # Wrap in LangChain HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=text_pipeline)
        logger.info("LLM setup complete")

    def setup_embeddings(self):
        """Initialize the custom Qwen embedding model."""
        logger.info("Setting up custom Qwen embeddings...")
        self.embeddings = CustomQwenEmbeddings()
        logger.info("Embeddings setup complete")

    def ingest_and_index_documents(self):
        """Load, chunk, and index all PDF documents."""
        logger.info(f"Loading PDFs from {self.source_dir}...")

        # Check if source directory exists
        if not os.path.exists(self.source_dir):
            os.makedirs(self.source_dir)
            logger.warning(f"Created empty source directory: {self.source_dir}")
            logger.warning("Please add PDF files to this directory and run again.")
            return False

        # Load PDFs
        loader = PyPDFDirectoryLoader(self.source_dir)
        documents = loader.load()

        if not documents:
            logger.warning(f"No PDF documents found in {self.source_dir}")
            return False

        logger.info(f"Loaded {len(documents)} pages from PDFs")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from documents")

        # Create vector store
        logger.info("Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        logger.info("Document indexing complete")
        return True

    def discover_topics(self, sample_size: int = 50) -> List[str]:
        """
        Dynamically discover main topics from the document collection.

        Args:
            sample_size: Number of documents to sample for topic discovery

        Returns:
            List of discovered topics
        """
        logger.info("Discovering topics from documents...")

        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": sample_size}
        )

        # Get a broad sample of documents
        sample_query = "main topics subjects themes content areas discussed"
        sample_docs = retriever.get_relevant_documents(sample_query)

        # Combine context
        context = "\n\n".join([doc.page_content for doc in sample_docs])

        # Create topic discovery prompt
        topic_prompt = PromptTemplate(
            input_variables=["context"],
            template="""Analyze the following context from multiple documents. Based ONLY on this context, identify the 5-7 most important, high-level topics that appear across the documents.

Context:
{context}

Instructions:
1. Look for recurring themes, subjects, and areas of focus
2. Identify broad topic categories that would encompass multiple related updates
3. Focus on substantive topics, not document structure elements
4. Base your response ONLY on the provided context

Respond ONLY with a JSON list of strings representing the main topics.
Example format: ["Project Budget", "Security Updates", "Timeline Changes", "Technical Architecture", "Team Structure"]

Topics:"""
        )

        # Create and run the chain using LCEL
        topic_chain = topic_prompt | self.llm | StrOutputParser()
        result = topic_chain.invoke({"context": context[:8000]}) # Limit context size

        # Parse the JSON response
        try:
            # Extract JSON from the response
            json_start = result.find('[')
            json_end = result.rfind(']') + 1
            if json_start != -1 and json_end != 0:
                json_str = result[json_start:json_end]
                topics = json.loads(json_str)
                logger.info(f"Discovered {len(topics)} topics: {topics}")
                return topics
            else:
                raise ValueError("No JSON list found in response")
        except Exception as e:
            logger.warning(f"Failed to parse topics: {e}")
            # Fallback to default topics
            default_topics = [
                "Project Status",
                "Technical Updates",
                "Budget and Resources",
                "Timeline and Milestones",
                "Risks and Issues"
            ]
            logger.info(f"Using default topics: {default_topics}")
            return default_topics

    def extract_topic_updates(self, topic: str) -> List[str]:
        """
        Extract all updates related to a specific topic from all documents.

        Args:
            topic: The topic to extract updates for

        Returns:
            List of unique updates/bullet points for the topic
        """
        logger.info(f"Extracting updates for topic: {topic}")

        # Create retriever with higher k value to get more comprehensive results
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 30}  # Get more documents for each topic
        )

        # Retrieve relevant documents for this topic
        retrieved_docs = retriever.get_relevant_documents(topic)

        if not retrieved_docs:
            logger.warning(f"No documents found for topic: {topic}")
            return []

        # Process documents in batches to avoid context length issues
        all_updates = []
        batch_size = 5

        for i in range(0, len(retrieved_docs), batch_size):
            batch_docs = retrieved_docs[i:i + batch_size]
            context = "\n\n".join([doc.page_content for doc in batch_docs])

            # Create extraction prompt
            extraction_prompt = PromptTemplate(
                input_variables=["topic", "context"],
                template="""Based ONLY on the provided context, extract ALL bullet points, updates, and key facts related to the topic: {topic}

Context:
{context}

Instructions:
1. Extract EVERY piece of information related to {topic}
2. Include all updates, changes, decisions, and facts
3. Each item should be a complete, standalone statement
4. Do NOT add information not present in the context
5. If no relevant information is found, respond with []

Respond ONLY with a JSON list of strings, where each string is a distinct update or fact.
Example: ["Update 1.1 - Budget increased by 10%", "Update 2.3 - Security review completed", "Update 5.7 - New team member joined"]

Updates:"""
            )

            # Create and run the extraction chain using LCEL
            extraction_chain = extraction_prompt | self.llm | StrOutputParser()
            result = extraction_chain.invoke({"topic": topic, "context": context[:6000]})

            # Parse the JSON response
            try:
                json_start = result.find('[')
                json_end = result.rfind(']') + 1
                if json_start != -1 and json_end != 0:
                    json_str = result[json_start:json_end]
                    batch_updates = json.loads(json_str)
                    all_updates.extend(batch_updates)
            except Exception as e:
                logger.warning(f"Failed to parse updates for batch {i//batch_size}: {e}")

        logger.info(f"Extracted {len(all_updates)} updates for {topic} (before deduplication)")
        return all_updates

    def deduplicate_updates(self, updates: List[str]) -> List[str]:
        """
        Remove duplicate and semantically similar updates.

        Args:
            updates: List of updates to deduplicate

        Returns:
            List of unique updates
        """
        if not updates:
            return []

        if len(updates) <= 10:
            # For small lists, use simple deduplication
            return list(dict.fromkeys(updates))

        logger.info(f"Deduplicating {len(updates)} updates...")

        # Process in batches for LLM-based deduplication
        batch_size = 20
        deduplicated = []

        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]

            # Create deduplication prompt
            dedup_prompt = PromptTemplate(
                input_variables=["updates"],
                template="""Review the following list of updates and consolidate semantically identical or highly similar items.

Updates:
{updates}

Instructions:
1. Keep only unique information
2. If two items describe the same update with slightly different wording, keep the most complete one
3. Preserve all distinct updates, even if they're related
4. Maintain the original meaning and specificity

Return ONLY a JSON list of the unique, consolidated updates.

Consolidated updates:"""
            )

            # Create and run the deduplication chain using LCEL
            dedup_chain = dedup_prompt | self.llm | StrOutputParser()
            updates_str = json.dumps(batch, indent=2)
            result = dedup_chain.invoke({"updates": updates_str})

            # Parse the JSON response
            try:
                json_start = result.find('[')
                json_end = result.rfind(']') + 1
                if json_start != -1 and json_end != 0:
                    json_str = result[json_start:json_end]
                    batch_deduped = json.loads(json_str)
                    deduplicated.extend(batch_deduped)
            except Exception as e:
                logger.warning(f"Failed to parse deduplicated batch: {e}")
                # Fall back to including all updates from this batch
                deduplicated.extend(batch)

        # Final pass to remove exact duplicates
        final_unique = list(dict.fromkeys(deduplicated))
        logger.info(f"Reduced from {len(updates)} to {len(final_unique)} unique updates")

        return final_unique

    def generate_report(self, compiled_data: Dict[str, List[str]], output_file: str = "Consolidated_Report.docx"):
        """
        Generate a DOCX report from the compiled data.

        Args:
            compiled_data: Dictionary mapping topics to lists of updates
            output_file: Name of the output file
        """
        logger.info(f"Generating report: {output_file}")

        # Create new document
        doc = Document()

        # Add title
        title = doc.add_heading("Consolidated Report - All Topics and Updates", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add executive summary
        doc.add_heading("Executive Summary", 1)
        summary = f"This report consolidates information from multiple PDF documents across {len(compiled_data)} main topics."
        doc.add_paragraph(summary)

        # Add a line break
        doc.add_paragraph()

        # Add each topic and its updates
        for topic, updates in compiled_data.items():
            # Add topic as heading
            doc.add_heading(topic, 1)

            if updates:
                # Add updates as bullet points
                for update in updates:
                    p = doc.add_paragraph(style='List Bullet')
                    p.add_run(update)
            else:
                doc.add_paragraph("No updates found for this topic.", style='Body Text')

            # Add spacing between topics
            doc.add_paragraph()

        # Add footer with generation info
        doc.add_page_break()
        doc.add_heading("Report Metadata", 2)
        doc.add_paragraph(f"Generated using RAG pipeline with Qwen3 models")
        doc.add_paragraph(f"Total topics processed: {len(compiled_data)}")
        total_updates = sum(len(updates) for updates in compiled_data.values())
        doc.add_paragraph(f"Total unique updates: {total_updates}")

        # Save document
        doc.save(output_file)
        logger.info(f"Report saved to {output_file}")

    def run_pipeline(self):
        """Execute the complete RAG pipeline."""
        logger.info("Starting RAG pipeline execution...")

        # Step 1: Setup models
        self.setup_embeddings()
        self.setup_llm()

        # Step 2: Ingest and index documents
        if not self.ingest_and_index_documents():
            logger.error("No documents to process. Exiting.")
            return

        # Step 3: Discover topics dynamically
        topics = self.discover_topics()

        # Step 4: Extract updates for each topic
        compiled_data = {}

        for topic in tqdm(topics, desc="Processing topics"):
            # Extract all updates for this topic
            topic_updates = self.extract_topic_updates(topic)

            # Deduplicate updates
            if topic_updates:
                unique_updates = self.deduplicate_updates(topic_updates)
                compiled_data[topic] = unique_updates
            else:
                compiled_data[topic] = []

            logger.info(f"Topic '{topic}': {len(compiled_data[topic])} unique updates")

        # Step 5: Generate the report
        self.generate_report(compiled_data)

        logger.info("Pipeline execution complete!")

        # Print summary
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Topics discovered: {len(topics)}")
        for topic, updates in compiled_data.items():
            print(f"  - {topic}: {len(updates)} updates")
        print(f"\nReport generated: Consolidated_Report.docx")
        print("="*50)


def main():
    """Main entry point for the script."""
    # Check for required directories
    source_dir = "./source_pdfs"

    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"Created {source_dir} directory.")
        print("Please add your PDF files to this directory and run the script again.")
        return

    # Check if there are PDFs in the directory
    pdf_files = list(Path(source_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        print("Please add PDF files to process and run the script again.")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    # Initialize and run pipeline
    pipeline = RAGPipeline(source_dir=source_dir)

    try:
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
