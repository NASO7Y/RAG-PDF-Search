# RAG-PDF-Search

## 📌 Overview
RAG-PDF-Search is a **Retrieval-Augmented Generation (RAG) application** built with **Streamlit**, designed to enable efficient searching within PDF documents. It utilizes **Ollama** and **DeepSeek R1** for intelligent query answering based on document content. The application supports Arabic text processing and provides concise, context-aware responses.

## 🚀 Features
- 📝 **Upload and search within PDFs**
- 🧠 **Uses DeepSeek R1 via Ollama for NLP**
- 🔍 **Retrieval using FAISS for vector-based search**
- 📚 **Semantic text splitting for better chunking**
- 🌍 **Supports Arabic text processing**
- ⚡ **Fast, interactive UI using Streamlit**

## 🛠️ Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed, along with the required dependencies.

### Clone the repository:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/RAG-PDF-Search.git
cd RAG-pdf-search
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the application:
```bash
streamlit run RAG.py
```

## 🏢 Tech Stack
- **Python**
- **Streamlit** (for UI)
- **LangChain**
- **Ollama & DeepSeek R1** (for language processing)
- **FAISS** (for vector search)
- **HuggingFace Embeddings**
- **PDFPlumber** (for document parsing)

## 📌 How It Works
1. Upload a **PDF file**.
2. The app processes and **embeds** the text.
3. Queries are matched to **relevant document sections**.
4. **Ollama & DeepSeek R1** generate concise responses.
5. Results are displayed in an interactive interface.

## 🤝 Contributions
Feel free to fork the repository and contribute!

