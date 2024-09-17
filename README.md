# targeted-case-RAG-System
# Description 
This project is a Streamlit-based web application designed to take queries and extract key information from multiple PDF documents. The system is for targeted-use cases, taking clinical research as an example. With this system, medical researchers can interact and get needed information from uploaded PDF documents. The system does this by reading through the uploaded PDF files and giving needed information as a result
# Tech Stack
Streamlit: For creating the interactive web application.
LangChain: For handling language models and conversational chains.
PyPDF2: For extracting text from PDF files.
Pinecone: For efficient vector store creation and retrieval.
HuggingFace: For embedding models used in the vector store.
Logging: To handle and record errors and evaluation results.
dotenv: To manage environment variables.
# Features
PDF Text Extraction: Extracts and processes text from multiple PDF documents.
Text Chunking: Splits the extracted text into manageable chunks.
Vector Store: Uses Pinecone for efficient text retrieval.
Conversational Retrieval: Allows users to ask questions about the content of the PDFs.
Evaluation Logging: Records evaluation results for the retrieval and generation components.
