Here is the comprehensive prompt for the agent.
This prompt is highly detailed to incorporate all your specific requirements: scalability for 100 PDFs, use of quantized models for local/efficient processing, strict anti-hallucination measures, and the dynamic topic discovery (Option B) pipeline.
🤖 Agent Prompt:
> System: You are an expert Python developer specializing in building scalable, local-first RAG (Retrieval-Augmented Generation) pipelines. Your task is to write a complete, executable Python script that performs topic-based extraction from a large number of PDFs and compiles them into a DOCX report.
> User:
> Goal:
> Create a Python script named auto_report.py that can process a large directory (e.S., 100+ PDFs) and generate a single, structured .docx report. The script must be optimized for local execution by using quantized models and efficient data handling.
> Core Pipeline (Mandatory):
>  * Ingest & Index: Efficiently load, chunk, and index all PDFs into a local vector store.
>  * Dynamic Topic Discovery: First, use the LLM to analyze the documents and dynamically generate a list of the main topics.
>  * Topic-Based Extraction: Loop through this dynamic topic list. For each topic, retrieve all relevant context from the vector store and use an LLM to extract all key bullet points.
>  * Deduplication: Clean the extracted bullet points to remove duplicates.
>  * Report Generation: Compile all unique bullet points, organized by their topic, into a .docx file.
> Technical Constraints (Strict):
> 1. Scalability (100+ PDFs):
>  * Do not try to load all document text into the LLM context. This must be a retrieval-based (RAG) pipeline.
>  * Use Chroma (or FAISS) for the vector store to handle the large number of indexed chunks.
> 2. Local & Quantized Models (Efficiency):
>  * Embeddings: Use HuggingFaceEmbeddings. Specify a lightweight, fast model (e.g., all-MiniLM-L6-v2 or bge-small-en-v1.5).
>  * LLM: Use LlamaCpp to run a local, quantized GGUF model. This is critical for performance and to respect the user's "lower precision" request.
>  * In the script, define the LlamaCpp LLM. Use a placeholder path like ./models/your_model.gguf and add a comment telling the user to download a model (e.g., Mistral-7B-Instruct-v0.2.Q5_K_M.gguf) and place it there.
> 3. Anti-Hallucination (Accuracy):
>  * All LLM prompts must be engineered to prevent hallucination.
>  * Rule 1: Explicitly instruct the LLM to "base its answer ONLY on the provided context."
>  * Rule 2: Instruct the LLM that if no relevant information is found in the context, it should output an empty list ([]) and nothing else.
> Detailed Script Implementation:
> Please use LangChain (LCEL) for all chains.
> Step 1: Document Ingestion & Indexing
>  * Use PyPDFDirectoryLoader (or iterate with PyPDFLoader) to load all PDFs from a ./source_pdfs/ directory.
>  * Chunk documents using RecursiveCharacterTextSplitter.
>  * Create the vector store using Chroma and the specified HuggingFaceEmbeddings model.
> Step 2: LLM & Chain Definitions
>  * Define the LlamaCpp LLM object, pointing to the placeholder model path.
>  * Define a ChatPromptTemplate and an StrOutputParser or JsonOutputParser as needed.
> Step 3: Dynamic Topic Generation (Pass 1)
>  * Create a retriever from the vector store.
>  * Retrieve a broad sample of documents (e.g., retriever.get_relevant_documents(query="all key topics")).
>  * Create a chain (topic_chain) with a prompt that instructs the local LLM to:
>    * "Analyze the following context. Based ONLY on this context, identify the 5-7 most important, high-level topics.
>    * "Respond ONLY with a JSON list of strings. Example: ['Project Budget', 'Security Risks', 'Timeline']"
>    * "If no topics are clear, respond with []."
>  * Run this chain to get the dynamic_topics_list.
> Step 4: Topic-Based Extraction Loop (Pass 2 & 3)
>  * Initialize an empty dictionary: compiled_data = {}.
>  * Loop through each topic in the dynamic_topics_list.
>  * Inside the loop:
>    * A. Retrieve: retrieved_docs = retriever.get_relevant_documents(f"All facts about {topic}").
>    * B. Extract Chain (Pass 2): Create a chain that takes context and topic. The prompt must be:
>      * "You are an extractor. Based ONLY on the provided context, extract all bullet points and key facts related to the topic: {topic}."
>      * "Respond ONLY with a JSON list of strings. Each string is one bullet point."
>      * "If no relevant information is found, respond ONLY with []."
>    * C. Deduplication Chain (Pass 3): Create a chain that takes the list of extracted bullets.
>      * Prompt: "Review the following list. Consolidate items that are semantically identical. Return ONLY the final, unique JSON list of strings."
>    * D. Store: Store the final, unique list in compiled_data[topic].
> Step 5: DOCX Generation
>  * Use python-docx.
>  * Create a new Document().
>  * Add a main title.
>  * Loop through the compiled_data dictionary:
>    * For each topic (key), add a document.add_heading(topic, level=1).
>    * For each bullet (in the value list), add a document.add_paragraph(bullet, style='List Bullet').
>  * Save the file as Consolidated_Report.docx.
> Final Output:
>  * The complete auto_report.py script.
>  * A requirements.txt file (must include langchain, llama-cpp-python, huggingface-hub, sentence-transformers, chromadb, pypdf, python-docx).
> 
