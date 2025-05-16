# End-to-End RAG System Implementation Using LangChain on a PDF Data Source

This project demonstrates an end-to-end implementation of a Retrieval-Augmented Generation (RAG) pipeline applied to user-uploaded PDF documents. It combines modern NLP techniques for document ingestion, table extraction, embedding generation, vector indexing, and contextual question answering using a local LLM.

## Features

* Upload and analyze any PDF document
* Extract raw text and optionally detect tabular data
* Chunk documents with customizable size and overlap
* Generate dense vector embeddings with SentenceTransformers
* Perform semantic similarity search using FAISS
* Answer natural language queries using LangChain's integration with the Mistral model via Ollama
* Interactive, user-friendly interface via Streamlit

## Architecture Overview

The pipeline follows these main steps:

1. Text and table extraction using `pdfplumber` and `camelot`
2. Text preprocessing and chunking using LangChain's `RecursiveCharacterTextSplitter`
3. Embedding generation using `SentenceTransformer`
4. Index creation with FAISS for fast vector similarity search
5. Query processing with relevant context retrieval and answer generation using LangChain + Ollama

## Technologies Used

| Category        | Tool/Library                |
| --------------- | --------------------------- |
| Backend         | Python                      |
| UI              | Streamlit                   |
| PDF Parsing     | pdfplumber, camelot         |
| Embeddings      | sentence-transformers       |
| Indexing        | FAISS                       |
| LLM Integration | LangChain, Ollama (Mistral) |

## Getting Started

### Prerequisites

Install dependencies with:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```text
streamlit
pdfplumber
camelot-py[cv]
sentence-transformers
faiss-cpu
langchain
langchain-community
```

Also ensure you have:

* Tesseract installed (if Camelot requires OCR)
* Ollama running locally with the Mistral model downloaded

### Running the Application

```bash
streamlit run main.py
```

The app will open in your default browser.

## Usage Instructions

1. Open the Streamlit UI.
2. Upload a PDF from the sidebar.
3. Configure chunk size, overlap, and enable/disable table extraction.
4. Click "Process Document" to generate embeddings and create the index.
5. Once processed, enter any natural language question about the document and receive an AI-generated answer along with source context.

## Code Structure

| File               | Description                                    |
| ------------------ | ---------------------------------------------- |
| `main.py`          | Streamlit UI logic and user interaction        |
| `rag_functions.py` | Core processing functions for the RAG pipeline |

### Key Functions in rag\_functions.py

* `extract_text_from_pdf(path)`: Extracts raw text using `pdfplumber`
* `extract_tables_from_pdf(path)`: Extracts tables using `camelot`
* `convert_tables_to_text(tables)`: Converts DataFrames to plain text
* `clean_and_split_text(text, chunk_size, chunk_overlap)`: Splits text into chunks
* `generate_embeddings(chunks)`: Converts text chunks into dense vectors
* `index_embeddings(embeddings)`: Creates a FAISS index
* `retrieve_documents(query, index, chunks)`: Fetches top-k relevant chunks
* `generate_response(prompt)`: Generates response using the Mistral model via Ollama
* `rag_pipeline(query, index, chunks)`: Orchestrates retrieval and generation

## Example Prompt

![PDF QA Demo](/Working.jpeg)
![PDF retrieved](/retrieved.jpeg)
![PDF Content](/Content_retrieved.jpeg)


