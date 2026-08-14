# AI Personal Document Semantic Search

An AI-powered document search and question-answering system designed to help users find information across their personal document collections using natural-language queries.

The project supports heterogeneous documents such as:

* PDF
* PPT/PPTX
* PNG/JPG/JPEG images

Instead of relying only on exact keyword matches, the system aims to understand the semantic meaning of a query and retrieve relevant content from indexed documents.

## Current Project Scope

The system is being developed around the following pipeline:

```text
Documents
   ↓
Text Extraction / OCR
   ↓
Text Chunking
   ↓
Embeddings
   ↓
Vector Index
   ↓
Semantic Search
   ↓
Relevant Results
```

An optional question-answering layer will use retrieved document content to generate grounded responses with source references.

## Technology

* Python
* FastAPI
* React
* Sentence Transformers
* FAISS
* SQLite
* PyMuPDF
* python-pptx
* Tesseract OCR

## Project Status

This project is currently under active development.

The initial implementation focuses on establishing a reliable document-ingestion and semantic-retrieval pipeline. Further development will improve document indexing, retrieval quality, evaluation, and overall system reliability.

## Objective

The long-term goal is to create a local-first personal document search system that can efficiently search large collections of heterogeneous documents using natural-language queries while preserving source information for retrieved content.
