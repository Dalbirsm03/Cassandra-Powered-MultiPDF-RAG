import os
import streamlit as st
from PyPDF2 import PdfReader
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Cassandra connection
ASTRA_DB_SECURE_BUNDLE_PATH = os.getenv("ASTRA_DB_SECURE_BUNDLE_PATH")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

cluster = Cluster(
    cloud={"secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH},
    auth_provider=PlainTextAuthProvider("token", ASTRA_DB_APPLICATION_TOKEN)
)
session = cluster.connect()
# Keyspace and table information
keyspace = "multiplepdf"
v_dimension = 768
table_name = "qa_mini_demo"

# Create the table and index
session.execute(f"""
    CREATE TABLE IF NOT EXISTS {keyspace}.{table_name} (
        id UUID PRIMARY KEY,
        text TEXT,
        vector VECTOR<FLOAT,{v_dimension}>
    );
""")

session.execute(f"""
    CREATE CUSTOM INDEX IF NOT EXISTS idx_{table_name}_vector
    ON {keyspace}.{table_name} (vector)
    USING 'StorageAttachedIndex'
    WITH OPTIONS = {{'similarity_function' : 'cosine'}};
""")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to convert text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate embeddings for text chunks using Google Generative AI
def generate_real_embeddings(text_chunks):
    model_name = 'models/embedding-001'
    model = genai.get_model(model_name)
    embeddings = []
    for chunk in text_chunks:
        embedding_response = genai.embed_content(model=model, content=chunk, task_type="retrieval_document")
        embeddings.append(embedding_response["embedding"])
    return embeddings

# Function to load text chunks into Cassandra
def load_data_into_cassandra(text_chunks):
    embeddings = generate_real_embeddings(text_chunks)
    for chunk, embedding in zip(text_chunks, embeddings):
        session.execute(
            f"INSERT INTO {keyspace}.{table_name} (id, text, vector) VALUES (%s, %s, %s)",
            (uuid4(), chunk, embedding)
        )


input_prompt = """You are an expert and professional in analyzing the output from PDFs. 
You have to format the sentences and complete the sentences but the meaning of the sentence should not be changed.
Do not mention the figure number and name in the output as the output is just in text and not in image format"""

# Function to perform similarity search
def user_input(user_question):
    model_name = 'models/embedding-001'
    model = genai.get_model(model_name)

    query_vector = genai.embed_content(model=model, content=user_question, task_type="retrieval_document")["embedding"]

    ann_query = f"""
        SELECT id, text, similarity_cosine(vector, {query_vector}) as sim
        FROM {keyspace}.{table_name}
        ORDER BY vector ANN OF {query_vector}
        LIMIT 1
    """
    results = session.execute(ann_query)
    responses = [row.text for row in results]
    return responses[0] if responses else "No relevant information found."

# Function to get the refined response using the generative model
def get_gemini_response(input_prompt, input_text):
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([input_prompt, input_text])
    return response.text

if "history" not in st.session_state:
    st.session_state["history"] = []

# Main function
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini")
    st.header("Chat with PDF using Gemini")

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if not pdf_docs:
                st.error("Please upload your PDF files first.")
            else:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                load_data_into_cassandra(text_chunks)
                st.success("PDF processed and text indexed!")
                
    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("Respond"):
        if user_question:
            st.session_state["history"].append(("YOU", user_question))
            response = user_input(user_question)
            final_response = get_gemini_response(input_prompt, response)
            st.session_state["history"].append(("", final_response))


    for sender, message in st.session_state["history"]:
        st.write(f"{sender} : {message}")

if __name__ == "__main__":
    main()
