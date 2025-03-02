# RAG-PDF-Search

## 📌 Overview
**RAG-PDF-Search** is a **Retrieval-Augmented Generation (RAG) application** built with **Streamlit**, designed for efficient searching within PDF documents. It leverages **Ollama** and **DeepSeek R1** to provide intelligent, context-aware responses based on document content. The application supports **Arabic text processing** and delivers concise answers to user queries.

## 🚀 Features
- 📂 **Upload and search within PDFs** effortlessly.
- 🧠 **AI-powered responses** using DeepSeek R1 via Ollama.
- 🔍 **Efficient document retrieval** with FAISS-based vector search.
- 📚 **Advanced text chunking** for better semantic understanding.
- 🌍 **Supports Arabic and multilingual processing**.
- ⚡ **Fast and interactive UI** using Streamlit.

## 🛠️ Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed, along with the required dependencies.

### Clone the repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/RAG-PDF-Search.git
cd RAG-PDF-Search
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the application
```bash
streamlit run RAG.py
```

## 🏢 Tech Stack
- **Python** (Core development language)
- **Streamlit** (User interface)
- **LangChain** (AI-powered retrieval and processing)
- **Ollama & DeepSeek R1** (Natural language processing)
- **FAISS** (Efficient vector-based search)
- **HuggingFace Embeddings** (Semantic text embeddings)
- **PDFPlumber** (PDF document parsing)

## 📌 How It Works
1. **Upload a PDF file** through the interface.
2. The application **processes and embeds** the text using HuggingFace embeddings.
3. Queries are matched to **relevant document sections** using FAISS retrieval.
4. **Ollama & DeepSeek R1** generate a concise, context-aware response.
5. The results are displayed in a **user-friendly Streamlit UI**.

## 🤝 Contributions
We welcome contributions! Feel free to fork the repository, submit issues, or create pull requests.

## 📬 Contact
For questions or feedback, feel free to open an issue or reach out to me:

- **GitHub:** [NASO7Y](https://github.com/NASO7Y)
- **Email:** ahmed.noshy2004@gmail.com
- **LinkedIn:** [Ahmed Noshy](https://www.linkedin.com/in/nos7y/)

---
⭐ If you find this project helpful, consider giving it a star is support😂🌹
