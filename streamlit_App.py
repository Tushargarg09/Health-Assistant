import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Health Assistant", layout="wide")

st.markdown("""
## Health Assistant: Your Personal Medical Report Assistant
Health Assistant is a sophisticated chatbot built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It analyzes patient-reported symptoms by processing the input data, creating a comprehensive understanding, and providing accurate medicine recommendations. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective healthcare experience.

### How It Works
Follow these simple steps to interact with Health Assistant:

1. **Enter Your API Key**: You'll need a Google API key for Health Assistant to access Google's Generative AI models. Obtain your API key here.

2. **Provide Your Symptoms**: Input your symptoms either by typing them in or uploading relevant medical documents. Health Assistant accepts multiple formats to ensure comprehensive analysis.

3. **Receive Medicine Recommendations**: After processing your symptoms, Health Assistant will suggest appropriate medicine options tailored to your condition for precise and effective treatment.
""")



api_key = st.text_input("Enter your Google API Key:", type="password", key="")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """You are an AI Medical Report Analyzer specialized in interpreting medical reports and laboratory results. When analyzing reports and answering questions, follow this protocol:
    
INITIALIZATION:

Load patient's medical report data
Set up medical data privacy mode
Prepare medical terminology reference system

ANALYSIS PROTOCOL:

REPORT CLASSIFICATION

Identify report type (blood work, imaging, pathology, etc.)
Note report date and time
Verify report completeness

PARAMETER EXTRACTION

Extract all measured parameters
Record reference ranges
Flag abnormal values
Note missing expected values

QUESTION ANALYSIS

Extract specific parameters requested
Identify time-based queries
Note any trend analysis requests
Check for comparison requests

DATA VALIDATION

Confirm parameter availability
Verify value ranges
Check result consistency
Identify data gaps

RESPONSE FORMULATION

Include relevant values with reference ranges
Highlight abnormal results
Explain medical terminology
Address all question components
Note unavailable requested data

RESPONSE GUIDELINES:
Always:

Use clear, accessible language
Include relevant numerical values
Explain medical terms
Note normal/abnormal status
Indicate missing information

Never:
Make diagnoses
Recommend treatments
Speculate beyond data
Compare to other patients
Provide medical advice

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain



def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    print("Context being used for the query:")
    print(context)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("Health Assistant👨‍⚕️")

    user_question = st.text_input("Ask a Question from the uploaded medical documents", key="user_question")

    if user_question and api_key:
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")

if __name__ == "__main__":
    main()
