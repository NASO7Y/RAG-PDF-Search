import io
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Set up Streamlit page
st.set_page_config(page_title="PDF Search using Ollama and DeepSeek R1", layout="wide")

st.title("🔍 Search within a PDF using Ollama and DeepSeek R1")

# Sidebar for file upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Sidebar retrieval settings
top_k = st.sidebar.slider("Number of retrieved chunks", min_value=1, max_value=10, value=4)

if uploaded_file is not None:
    # Load PDF without saving to disk
    pdf_bytes = io.BytesIO(uploaded_file.getvalue())
    loader = PDFPlumberLoader(pdf_bytes)
    docs = loader.load()

    # Splitting text for better search
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # More efficient model
    text_splitter = SemanticChunker(embedder)
    documents = text_splitter.split_documents(docs)

    # Vector store for efficient retrieval
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="mmr", search_kwargs={"k": top_k})

    # Define LLM
    llm = Ollama(model="deepseek-r1")

    # Prompt for query processing
    prompt = """
    Use the provided context to answer the question concisely in 3-4 sentences.
    If the answer is unknown, respond with "I don't know."
    
    Context: {context}
    Question: {question}
    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # LLM processing chain
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)

    # Retrieval & answering pipeline
    qa = RetrievalQA(
        llm_chain=llm_chain,
        retriever=retriever,
        return_source_documents=True
    )

    # User query
    user_input = st.chat_input("Ask a question about the PDF:")

    if user_input:
        with st.spinner("Processing..."):
            response = qa(user_input)
            st.subheader("Answer:")
            st.write(response["result"])

            # Display sources
            st.subheader("Sources:")
            for doc in response["source_documents"]:
                st.write(f"- {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")
else:
    st.write("📂 Please upload a PDF to start searching.")
