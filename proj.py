# Import necessary libraries
import os, getpass, time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# PDF and document readers
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
# Document splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embedding model
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# LLM
from langchain_groq import ChatGroq
# Vector database
import pinecone
from langchain_pinecone import PineconeVectorStore
# Chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from langchain.docstore.document import Document 
from sentence_transformers import SentenceTransformer  
from pydantic import BaseModel

from evaluation import evaluate_retrieval, evaluate_model, time_execution
from logging_utils import log_and_evaluate
# Load environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
from model import *

# Embedding setup
index_name = 'aisoc-rag'
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

models = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

embeddings = huggingface_embeddings

class YourModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

# Function to handle chat history
def update_chat_history(query, response):
    # Check if the session state has a 'chat_history' list, if not, initialize it
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Append the new query and response to the chat history
    st.session_state['chat_history'].append({"query": query, "response": response})

# Function to display chat history
def display_chat_history():
    if 'chat_history' in st.session_state and st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.markdown(f"**You:** {chat['query']}")
            st.markdown(f"**Assistant:** {chat['response']}")
            st.markdown("---")

# Main function
def main():
    st.title("Clinical Research Paper Assistant")

    # Display chat history first
    display_chat_history()

    # Sidebar: Upload Documents
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Upload your clinical research papers (PDF only)", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        st.sidebar.write(f"Uploaded {len(uploaded_files)} file(s).")

        docs_before_split = []
        for uploaded_file in uploaded_files:
            # Extract text from the uploaded PDF file
            def extract_text_from_pdf(uploaded_file):
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text

            text = extract_text_from_pdf(uploaded_file)
            document = Document(page_content=text, metadata={"source": uploaded_file.name})
            docs_before_split.append(document)

        # Split the documents into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        documents = text_splitter.split_documents(docs_before_split)

        for uploaded_file in uploaded_files:
            st.sidebar.success(f"Processed file: {uploaded_file.name}")

        # Ask a Question
        st.header("Ask a Question")
        selected_model = st.selectbox("Select your preferred model:", models)

        llm = ChatGroq(
            model=selected_model,
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=5,
        )

        prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
        1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
        2. If you find the answer, write the answer in a concise way with five sentences maximum.

        {context}

        Question: {question}

        Helpful Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        index_name = "aisoc-rag"
        vectordb = PineconeVectorStore.from_documents(
            documents,
            embedding=embeddings,
            index_name=index_name
        )

        query = st.text_input("Enter your question about the uploaded documents:")

        # Function to get retrieval chain
        def get_retrieval_chain(vectordb, selected_model, prompt):
            retrieval_qa = RetrievalQA.from_chain_type(
                llm=selected_model,
                chain_type="stuff",
                retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            return retrieval_qa

        # Function to query the retrieval QA
        def query_retrieval_qa(retrieval_qa, query):
            result = retrieval_qa.invoke({"query": query})
            return result['result']

        if query:
            
            retrieval_qa_instance = get_retrieval_chain(vectordb, llm, PROMPT)
            
            retrieved_result, retrieval_time = time_execution(retrieval_qa_instance.invoke, {"query": query})

            relevant_docs = ["doc1", "doc2"]  # Dummy relevant documents for evaluation
            retrieved_docs = [doc.metadata["source"] for doc in retrieved_result["source_documents"]]

            query_result = retrieved_result["result"]
            expected_response = "Some expected answer"  # Replace with actual expected answer

            _, response_time = time_execution(lambda: query_result)

            st.write("Result:", query_result)

            log_and_evaluate(query, retrieved_docs, relevant_docs, query_result, expected_response, retrieval_time, response_time)
            update_chat_history(query, query_result)

    with st.form(key='input_form', clear_on_submit=True):
        submit_button = st.form_submit_button(label="Send")

if __name__ == "__main__":
    main()
