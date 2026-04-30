import streamlit as st
import os
import json
import tempfile

from Loaders.loaders import load_pdf
from Loaders.webbase import load_url
from Loaders.img import load_image
from Processing.chunking import chunk_documents
from vectorstore.vectordb import create_vectorstore, retrieve_docs
from rag_pipeline import generate_answer


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Multimodal Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>
    .main {
        background-color: #0f1117;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    .title-text {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
        color: #ffffff;
    }

    .subtitle-text {
        text-align: center;
        font-size: 16px;
        color: #a1a1aa;
        margin-bottom: 30px;
    }

    .upload-box {
        border: 1px solid #2d2f3a;
        border-radius: 18px;
        padding: 18px;
        background-color: #171923;
        margin-bottom: 20px;
    }

    .source-box {
        background-color: #171923;
        border: 1px solid #2d2f3a;
        border-radius: 14px;
        padding: 12px;
        margin-top: 12px;
        font-size: 14px;
    }

    div[data-testid="stChatInput"] {
        border-radius: 20px;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# MEMORY
# =========================

MEMORY_FILE = "chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_memory(chat_history):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=4)


# =========================
# SESSION STATE
# =========================

if "db" not in st.session_state:
    st.session_state.db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_memory()

if "show_upload_options" not in st.session_state:
    st.session_state.show_upload_options = False

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# =========================
# HELPER FUNCTION
# =========================

def add_knowledge(docs):
    chunks = chunk_documents(docs)

    if st.session_state.db is None:
        st.session_state.db = create_vectorstore(chunks)
    else:
        st.session_state.db.add_documents(chunks)


def save_uploaded_file(uploaded_file, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name


# =========================
# HEADER
# =========================

st.markdown('<div class="title-text">Aagrah Multimodal Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Chat with PDFs, URLs, and images using RAG, citations, and memory.</div>',
    unsafe_allow_html=True
)


# =========================
# TOP CONTROLS
# =========================

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🧹 Clear Knowledge"):
        st.session_state.db = None
        st.session_state.last_sources = []
        st.success("Knowledge cleared.")

with col2:
    if st.button("🧠 Clear Memory"):
        st.session_state.chat_history = []
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        st.success("Memory cleared.")

with col3:
    if st.session_state.db is not None:
        st.success("RAG Active")
    else:
        st.info("Normal Chat Mode")


# =========================
# PLUS UPLOAD OPTION
# =========================

st.markdown("")

plus_col, text_col = st.columns([0.08, 0.92])

with plus_col:
    if st.button("➕"):
        st.session_state.show_upload_options = not st.session_state.show_upload_options

with text_col:
    st.markdown("##### Add knowledge source")


if st.session_state.show_upload_options:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    upload_type = st.radio(
        "Choose source type",
        ["PDF", "URL", "Image"],
        horizontal=True
    )

    if upload_type == "PDF":
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded_pdf is not None:
            if st.button("Load PDF"):
                try:
                    pdf_path = save_uploaded_file(uploaded_pdf, ".pdf")
                    docs = load_pdf(pdf_path)
                    add_knowledge(docs)
                    st.success("PDF loaded successfully.")
                except Exception as e:
                    st.error(f"Error loading PDF: {e}")

    elif upload_type == "URL":
        url = st.text_input("Enter website URL")

        if st.button("Load URL"):
            if url.strip():
                try:
                    docs = load_url(url)
                    add_knowledge(docs)
                    st.success("URL loaded successfully.")
                except Exception as e:
                    st.error(f"Error loading URL: {e}")
            else:
                st.warning("Please enter a URL.")

    elif upload_type == "Image":
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        if uploaded_image is not None:
            if st.button("Load Image"):
                try:
                    image_path = save_uploaded_file(uploaded_image, ".png")
                    docs = load_image(image_path)
                    add_knowledge(docs)
                    st.success("Image loaded successfully.")
                except Exception as e:
                    st.error(f"Error loading image: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# CHAT HISTORY DISPLAY
# =========================

for msg in st.session_state.chat_history:
    role = msg["role"]

    with st.chat_message(role):
        st.write(msg["content"])


# =========================
# CHAT INPUT
# =========================

query = st.chat_input("Ask anything...")

if query:
    with st.chat_message("user"):
        st.write(query)

    try:
        if st.session_state.db is not None:
            
            with st.spinner("Thinking..."):
                retrieved_docs = retrieve_docs(st.session_state.db, query)

                # IMPORTANT:
                # Positional arguments used to avoid keyword mismatch errors
                answer = generate_answer(
                    retrieved_docs,
                    query,
                    st.session_state.chat_history[-20:]
                )

            with st.chat_message("assistant"):
                st.write(answer)

                st.markdown("#### 📚 Sources")
                for i, doc in enumerate(retrieved_docs, start=1):
                    source = doc.metadata.get("source", "Unknown source")
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(
                        f"""
                        <div class="source-box">
                        <b>{i}. Source:</b> {source}<br>
                        <b>Page:</b> {page}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        else:
            retrieved_docs = []
            with st.spinner("Thinking..."):

                answer = generate_answer(
                    [],
                    query,
                    st.session_state.chat_history[-6:]
                )

            with st.chat_message("assistant"):
                st.write(answer)

        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.chat_history = st.session_state.chat_history[-6:]
        save_memory(st.session_state.chat_history)

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error generating answer: {e}")