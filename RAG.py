import streamlit as st
import io
from arabic_support import support_arabic_text
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set up the app interface
st.set_page_config(page_title="Search PDF with Ollama & DeepSeek", layout="wide")
support_arabic_text(all=True)

st.title("Search within a PDF using Ollama and DeepSeek")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Load the PDF directly from memory (avoid unnecessary disk writes)
        pdf_bytes = io.BytesIO(uploaded_file.getvalue())
        loader = PDFPlumberLoader(pdf_bytes)
        docs = loader.load()

        # Check if document is empty
        if not docs:
            st.error("The uploaded PDF is empty or unreadable. Please try another file.")
            st.stop()

        # Initialize embedding model
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Adaptive chunk size for optimal performance
        chunk_size = max(500, min(1000, len(docs) // 10))  
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5,
            separators=["\n\n", "\n", " ", ""]
        )

        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embedder)
        retriever = vector.as_retriever(search_kwargs={"k": 4})

        # Initialize LLM
        llm = OllamaLLM(model="deepseek-r1")

        # Define prompt template
        template = """
        Use the following context to answer the question concisely in 3-4 sentences.
        If the answer is unknown, respond with "I don’t know" without making anything up.
        
        Context: {context}   
        Question: {question}  
        Answer:
        """
        prompt = PromptTemplate.from_template(template)

        # Chain for retrieval-augmented generation (RAG)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    # Interactive chat interface
    st.subheader("Ask a question about the document:")
    user_input = st.chat_input("Type your question here...")

    if user_input:
        with st.spinner("Retrieving answer..."):
            response = rag_chain.invoke(user_input)
            relevant_chunks = retriever.get_relevant_documents(user_input)

            # Display retrieved context
            st.markdown("### Retrieved Context:")
            for idx, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {idx+1}:** {chunk.page_content[:500]}...")  # Show snippet

            # Display AI-generated response
            st.markdown("### AI Answer:")
            st.write(response)
else:
    st.info("Please upload a PDF file to begin.")
