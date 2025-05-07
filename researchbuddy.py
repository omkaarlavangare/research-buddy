# Import necessary libraries
import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()

# Set API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Streamlit app
st.title("Research Buddy")

st.markdown("Research Buddy helps you with your research by answering your questions based on your research materials.\n\n"
            "To start using this system first load your research materials by clicking on \"Load research directory\".\n\n"
            "Once the directory is loaded you can start asking questions. All the best!!!")

# Load Language Model
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def load_vector_embeddings():
    """Load and create vector embeddings if not already in session state."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Manually load all PDFs in directory
        pdf_files = glob.glob("./knowledgebase/*.pdf")
        all_docs = []
        for pdf in pdf_files:
            loader = PyPDFLoader(pdf)
            all_docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = text_splitter.split_documents(all_docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Research directory loaded successfully!")



# Button to load research documents
if st.button("Load research directory"):
    with st.spinner("Loading your research materials into knowledge base...Please wait"):
        load_vector_embeddings()

user_query = st.text_input("How can I help with your research")

# Processing user query
if st.button("Search"):
    if user_query and "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        
        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': user_query})
        response_time = time.process_time() - start_time
            
        st.write("**Response:**", response.get('answer', "No answer found."))
        st.write(f"Response Time: {response_time:.2f} seconds")
    
    # Display retrieved document excerpts
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(f"**Document {i+1}:**")
            st.write(doc.page_content)
            st.write("---")

