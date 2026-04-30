import streamlit as st

from Loaders.loaders import load_pdf
from Loaders.webbase import load_url
from Loaders.img import load_image
from Processing.chunking import chunk_documents
from vectorstore.vectordb import create_vectorstore, retrieve_docs
from rag_pipeline import generate_answer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

api_key = os.getenv("GEMINI_API_KEY")

# Debug check
st.write("ENV path:", ENV_PATH)
st.write("ENV exists:", ENV_PATH.exists())
st.write("API key loaded:", bool(api_key))

if not api_key:
    st.error("GEMINI_API_KEY not found. Check .env file.")
    st.stop()
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key
)

# -------------------------
# SESSION STATE (memory)
# -------------------------
if "db" not in st.session_state:
    st.session_state.db = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------
# UI TITLE
# -------------------------
st.title("🧠 Multimodal RAG Assistant")

with st.sidebar:
    st.header("📚 Load Knowledge")

    source_type = st.radio("Choose source", ["PDF", "URL", "Image"])

    docs = None

    if source_type == "PDF":
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

        if uploaded_pdf is not None and st.button("Add PDF to Knowledge"):
            temp_path = BASE_DIR / uploaded_pdf.name

            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            docs = load_pdf(str(temp_path))

    elif source_type == "URL":
        url = st.text_input("Enter website URL", key="url_input")

        if url and st.button("Add URL to Knowledge"):
            docs = load_url(url)

    elif source_type == "Image":
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="image_uploader")

        if uploaded_image is not None and st.button("Add Image to Knowledge"):
            temp_path = BASE_DIR / uploaded_image.name

            with open(temp_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            docs = load_image(str(temp_path))

    if docs:
        chunks = chunk_documents(docs)

        if st.session_state.db is None:
            st.session_state.db = create_vectorstore(chunks)
        else:
            st.session_state.db.add_documents(chunks)

        st.success("Knowledge added ✅ Now ask questions in chat.")

# -------------------------
# CHAT UI
# -------------------------

user_input = st.chat_input("Ask me anything...", key="main_chat_input")

if user_input:
    st.chat_message("user").write(user_input)

    if st.session_state.db is None:
        response = llm.invoke(user_input).content
    else:
        docs = retrieve_docs(st.session_state.db, user_input)
        response = generate_answer(docs,user_input,llm)

    st.chat_message("assistant").write(response)