# DocChat: Advanced Agentic RAG & Document Intelligence Assistant

An enterprise-grade, multi-agent dynamic information retrieval system built with **LangGraph**, **Docling**, and **ChromaDB**, featuring automated self-correction mechanisms to eliminate AI hallucinations and accurately parse complex unstructured documents (tables, figures, and dense text).

---

## 🚀 Key Features & Architecture
* **Layout-Aware Parsing (Docling):** Extracts clean hierarchical text and table structures from complex PDFs.
* **Hybrid Retrieval (BM25 + ChromaDB):** Combines sparse keyword search with dense vector embeddings for high-precision context recall.
* **Multi-Agent Orchestration (LangGraph):** Coordinates specialized Retriever, Research, and Verification agents in a stateful workflow.
* **Self-Correction & Fallback Routing:** Automatically detects potential hallucinations, verifies factual grounding, and handles out-of-scope queries.
* **Interactive UI (Gradio):** Provides real-time document querying and validation feedback.

---

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Orchestration:** LangGraph, LangChain
* **Parsing & Search:** Docling, ChromaDB, Rank-BM25, Sentence Transformers
* **Interface:** Gradio

---

## 📂 Project Structure
```text
DocChat/
├── app.py                  # Gradio web interface & execution pipeline
├── agent_workflow.py       # LangGraph multi-agent logic & self-correction loop
├── retriever_builder.py    # Hybrid search implementation (BM25 + ChromaDB)
├── document_processor.py   # PDF parsing and chunking configuration using Docling
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
