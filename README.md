# DocChat: Advanced Agentic RAG & Document Intelligence Assistant

An enterprise-grade, multi-agent dynamic information retrieval system built from scratch using **LangGraph**, **Docling**, and **ChromaDB**. Designed with automated self-correction mechanisms to eliminate AI hallucinations and accurately parse complex unstructured documents (tables, figures, and dense text).

---

## 🚀 Project Highlights & Architecture
Traditional Retrieval-Augmented Generation (RAG) pipelines often suffer from blind retrieval, failure to handle structured tables, and hallucinated answers. **DocChat** solves these challenges by implementing a stateful multi-agent workflow:

1. **Document Intelligence (Docling):** Extracts clean hierarchical text, metadata, and tables from complex PDF documents.
2. **Hybrid Retriever:** Combines sparse keyword search (**BM25**) with dense vector embeddings (**ChromaDB**) for high-precision recall.
3. **Research Agent:** Analyzes the retrieved context and synthesizes structured, fact-based answers.
4. **Verification & Self-Correction Agent:** Cross-checks the generated output against the source document chunks. If inaccuracies or hallucinations are detected, the system automatically triggers a self-correction loop to refine the answer.

---

## 🛠️ Tech Stack
* **Programming Language:** Python 3.10+
* **Orchestration & Workflow:** LangGraph, LangChain
* **Document Parser:** Docling
* **Vector Database & Search:** ChromaDB, BM25 / Rank-BM25
* **User Interface:** Gradio

---

## 📂 Project Structure
```text
DocChat/
├── app.py                  # Main application script & Gradio UI interface
├── agent_workflow.py       # LangGraph state graph and multi-agent logic
├── retriever.py            # Hybrid search implementation (BM25 + ChromaDB)
├── document_parser.py      # PDF parsing and chunking configuration using Docling
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
