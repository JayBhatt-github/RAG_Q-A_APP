import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configure Google Generative AI
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    pdf_reader = PdfReader(pdf_docs)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and store FAISS index in session state
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    st.session_state["vector_store"] = vector_store

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer questions from the provided context and provide answers to the best of your knowledge. 
    If the answer is not known, say, "I can't answer this question." Do not provide incorrect answers.
    \n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input
def user_input(user_question):
    vector_store = st.session_state["vector_store"]
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="RAG CHAT APP")
    st.header("RAG Q&A with Gemini 📄")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("📂 Upload PDF", type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Done processing!")
        # Contact section
        st.subheader("Contact Me")
        st.markdown("""
            <div style="display: flex; align-items: center; background-color: #f9f9f9; padding: 10px; border-radius: 5px;">
                <a href="https://github.com/yourusername" target="_blank" style="margin-right: 20px; text-decoration: none; color: black;">
                    <img src="https://img.icons8.com/ios-filled/50/000000/github.png" style="filter: invert(100%); width: 30px; height: 30px; margin-right: 5px;"/>
                    GitHub
                </a>
                <a href="https://www.linkedin.com/in/yourusername/" target="_blank" style="margin-right: 20px; text-decoration: none; color: black;">
                    <img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" style="filter: invert(100%); width: 30px; height: 30px; margin-right: 5px;"/>
                    LinkedIn
                </a>
                <a href="mailto:youremail@example.com" target="_blank" style="text-decoration: none; color: black;">
                    <img src="https://img.icons8.com/ios-filled/50/000000/email.png" style="filter: invert(100%); width: 30px; height: 30px; margin-right: 5px;"/>
                    Email
                </a>
            </div>
        """, unsafe_allow_html=True)

    # Input field for user question
    user_question = st.text_input("Enter your question")
    if user_question and "vector_store" in st.session_state:
        user_input(user_question)
    elif user_question:
        st.error("Please upload and process a PDF first!")

# Run the Streamlit app
if __name__ == "__main__":
    main()
