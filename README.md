# CV RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF resume/CV and ask questions about it.

Powered by:
- **LLM**: Groq (Llama-3.1-8B-Instant) for fast inference
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (local)
- **Vector Store**: FAISS
- **Framework**: LangChain

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j4UcNNr7c8oARpJ2JGfufvKoG6ft4mrD?usp=sharing)

## How It Works
1. Upload a PDF (e.g., a resume).
2. The text is extracted, chunked, embedded, and indexed in FAISS.
3. Ask questions – the bot retrieves relevant chunks and answers using Groq's LLM.
4. Strict prompt ensures it only uses info from the PDF.

## Quick Start in Colab
Click the badge above to open and run directly in Google Colab.

## Local Setup
```bash
git clone https://github.com/muhammadaliqamar/cv-rag-chatbot.git
cd cv-rag-chatbot
pip install -r requirements.txt
