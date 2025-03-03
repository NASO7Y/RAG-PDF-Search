# RAG-PDF-Search

## 📌 Overview
**RAG-PDF-Search** is a **Retrieval-Augmented Generation (RAG) application** built with **Streamlit**, designed for intelligent searching within PDF documents. By leveraging **Ollama** and **DeepSeek R1**, it provides highly relevant, context-aware responses based on document content. The application also supports **Arabic text processing** and delivers concise answers to user queries.

## 🚀 Features
- 📂 **Upload and search within PDFs** with ease.
- 🧠 **AI-powered responses** using DeepSeek R1 via Ollama.
- 🔍 **Efficient document retrieval** powered by FAISS-based vector search.
- 📚 **Advanced text chunking** for enhanced semantic understanding.
- 🌍 **Multilingual and Arabic text support** for diverse use cases.
- ⚡ **Fast and interactive UI** built with Streamlit.

## 🛠️ Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed and all required dependencies.

### Clone the Repository
```bash
git clone https://github.com/NASO7Y/RAG-PDF-Search.git
cd RAG-PDF-Search
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run RAG.py
```

## 🏢 Tech Stack
- **Python** (Core development language)
- **Streamlit** (User interface framework)
- **LangChain** (AI-powered retrieval and processing)
- **Ollama & DeepSeek R1** (Natural language processing models)
- **FAISS** (Fast vector-based search)
- **HuggingFace Embeddings** (Semantic text embeddings)
- **PDFPlumber** (PDF document processing)

## 📌 How It Works
1. **Upload a PDF file** via the Streamlit interface.
2. The application **extracts, processes, and embeds** the text using HuggingFace embeddings.
3. Queries are matched to **relevant document segments** using FAISS-based retrieval.
4. **Ollama & DeepSeek R1** generate a precise, context-aware response.
5. The results are displayed in a **user-friendly Streamlit UI**.

## 🤝 Contributions
We welcome all contributions! Feel free to fork the repository, submit issues, or create pull requests.

## 📬 Contact
For any questions or feedback, feel free to reach out:

- **GitHub:** [NASO7Y](https://github.com/NASO7Y)
- **Email:** ahmed.noshy2004@gmail.com
- **LinkedIn:** [Ahmed Noshy](https://www.linkedin.com/in/nos7y/)


---
⭐ If you find this project helpful, consider giving it a star is support😂🌹
