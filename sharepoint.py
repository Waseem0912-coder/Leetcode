You got it. First, here is the detailed Pydantic Schema and LangChain structure for the action item extraction. Then, I'll provide the complete prompt for the agent.
1. 🏗️ Pydantic Schema & LangChain Structure
This is the key to getting reliable, structured JSON data from the LLM.
The Pydantic Schema
You'll define a Python class that tells the LLM exactly what data structure you want back. This gives it "guardrails."
from pydantic import BaseModel, Field
from typing import List, Optional

class ActionItem(BaseModel):
    """A single, discrete action item or task identified from the text."""
    
    description: str = Field(
        ..., 
        description="The full text of the action item. Must be a specific, actionable task."
    )
    
    source_document: str = Field(
        ..., 
        description="The name of the source PDF document where this action item was found."
    )
    
    context: Optional[str] = Field(
        default=None, 
        description="Brief context or project associated with the action item, if available."
    )

class ActionItemList(BaseModel):
    """A list of all unique action items extracted from the reports."""
    
    action_items: List[ActionItem]

The LangChain Structure (using LCEL)
You would use this ActionItemList schema with your LLM's .with_structured_output() method. This forces the LLM to only output JSON that validates against your schema.
# This is a conceptual example of the chain part
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialize your LLM
# Make sure to use a model that supports function calling / structured output
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# 2. Bind the schema to the LLM
structured_llm = llm.with_structured_output(ActionItemList)

# 3. Create a prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant. Your task is to analyze the provided text
     from multiple reports and extract ALL action items.
     
     An action item is a discrete task or to-do. Ignore vague statements.
     Use the 'source_document' field from the metadata to populate the source.
     
     Respond ONLY with the JSON structure requested."""),
    ("user", "{retrieved_context}") # This context comes from your vector store
])

# 4. Create the chain
# This chain will automatically retrieve context, format the prompt,
# call the LLM, and parse the output into your Pydantic object.
action_item_chain = (
    retriever | 
    (lambda docs: {"retrieved_context": format_docs(docs)}) | # 'format_docs' is a helper function you write
    prompt |
    structured_llm
)

# 5. Run the chain
# The 'result' will be an instance of your ActionItemList Pydantic class
# result = action_item_chain.invoke({"query": "Extract all action items"})

2. 🤖 Agent Prompt to Generate the Script
Here is a comprehensive prompt you can give to an AI agent (like me, in a new chat) to generate the complete Python script, detailing all the steps we've discussed.
> System: You are an expert Python developer specializing in building RAG (Retrieval-Augmented Generation) pipelines using LangChain and LlamaIndex. Your task is to write a complete, executable Python script based on the user's requirements.
> User:
> Goal:
> Create a Python script named report_generator.py. This script will function as an "Auto Report Generator" that reads multiple PDF files from a directory, uses a RAG pipeline to process them, and generates a single consolidated PDF report as output.
> Output Requirements:
> The final generated PDF must contain two distinct sections:
>  * Consolidated Report: A comprehensive summary and consolidation of the key information, findings, and statuses from all the input PDFs.
>  * Unique Action Items: A clearly formatted list (e.g., a bulleted list or table) of all unique action items extracted from the documents. This list must be deduplicated.
> Script Implementation Details:
> Please use LangChain (LCEL) for the pipeline orchestration.
> Step 1: Document Ingestion & Indexing
>  * The script must accept a directory path (e.g., ./source_pdfs) as input.
>  * Load PDFs: Use UnstructuredPDFLoader (or LlamaParse if you note it as a requirement) to load all PDFs from the directory. It's crucial to handle complex layouts, tables, and lists.
>  * Chunking: Use a RecursiveCharacterTextSplitter to chunk the documents. Ensure the splitting is context-aware (e.g., splitting on markdown headers or new lines).
>  * Embeddings: Use a local embedding model (e.g., HuggingFaceEmbeddings with all-MiniLM-L6-v2) to create vector embeddings.
>  * Vector Store: Store the chunks and embeddings in a local vector store like Chroma or FAISS.
>  * Retriever: Create a retriever object from the vector store.
> Step 2: The RAG Pipeline (Two-Pass Generation)
> This is a multi-step process.
>  * Pass 1: Consolidated Report Generation
>    * Create a chain that uses the retriever to fetch relevant context from all documents based on a general query (e.g., "Consolidate all report findings").
>    * Use a prompt template (e.g., ChatPromptTemplate) to instruct an LLM (e.g., ChatOpenAI) to write a full, consolidated report in Markdown format.
>  * Pass 2: Unique Action Item Extraction (Two-Step Extraction)
>    * A. Initial Extraction (Structured Output):
>      * Define a Pydantic schema for the action items. It must include fields for description: str and source_document: str.
>      * Create a new chain that passes the retrieved context to an LLM using .with_structured_output() bound to the Pydantic schema. This will extract all action items it can find, resulting in a JSON list (which may have duplicates).
>    * B. Deduplication:
>      * Take the raw JSON list of action items from step (A).
>      * Create a second LLM chain. The prompt for this chain should be: "Review the following JSON list of action items. Consolidate items that are semantically identical or refer to the same task. Output only the final, unique list of action items, in the same JSON format."
> Step 3: Final PDF Generation
>  * The script must take the Markdown text from Pass 1 and the final unique JSON from Pass 2.
>  * Format the JSON action items into a clean Markdown list or table.
>  * Append the "Action Items" Markdown to the "Consolidated Report" Markdown.
>  * Use a Python library (like markdown-pdf, WeasyPrint, or ReportLab) to convert this final, complete Markdown string into a single output PDF file (e.g., Consolidated_Report.pdf).
> Please include requirements.txt and provide comments in the code explaining each major step (Loading, Indexing, RAG Chain 1, RAG Chain 2, PDF Generation).
> 
