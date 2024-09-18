import streamlit as st
import os
import groq
from pydantic import BaseModel
from dotenv import load_dotenv
import traceback
# load vector database
import pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate
#from app import get_retrieval_chain

load_dotenv()



class chat_bot():
    def __init__(self):
    # Set GRO_API_KEY = "your api key" in the .env file, then load it below
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        # Define embedding model
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # Run generative search otherwise

        query:str
        output:str = ""
        self.models = [
        # "llama-3.1-405b-reasoning",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
        self.output_type = ["Stream", "Batch"]
        self.token_class = { "short":150, "Moderate":700, "Long": 1536}
        
        prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

Helpful Answer:
PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context", "question"]
"""
        sys_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


  
          
    def get_response(self, message, token, model="llama-3.1-70b-versatile", temperature=0):
        try:            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{self.sys_prompt}"},
                    {"role": "user", "content": f"{message}"}
                ],
                stream=True,
                temperature=temperature,
                max_tokens= token,
            )
            return response
    
        except Exception as e:
            print(traceback.format_exc())
            return {
                "error": str(e),
                "status_code": 400
            }


    def get_response_batch(self, message, token, model="llama-3.1-70b-versatile", temperature=0):
        try:
            response = self.client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "system", "content": f"{self.sys_prompt}"},
                    {"role": "user", "content": message},
                ],
                response_format = {"type": "text"},
                temperature = temperature,
                max_tokens=token
            )
            return response
    
        except Exception as e:
            print(traceback.format_exc())
            return {
                "error": str(e),
                "status_code": 400
            }
    def init_vector_store(self, documents):
        # Initialize Pinecone vector store with documents
        self.vector_store = PineconeVectorStore.from_documents(
            documents,
            index_name='aisoc-rag',
            embedding=self.embedding_model
        )

    def get_retrieval_qa_chain(self, retriever, selected_model, prompt_template):
        llm = ChatGroq(
        model=selected_model,
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=5,
    )
        # Return a RetrievalQA chain instance
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        st.session_state.chat_active = True
        return retrieval_chain

    #def get_answer(query):
    # Get retrieval chain
        retrieval_chain = get_retrieval_chain(st.session_state.vector_store)
    # Get answer from retrieval chain
        answer = retrieval_chain({"query": query})
    
        return answer 
