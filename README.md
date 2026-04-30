# Multimodal-Assistant

# Aagrah Multimodal Assistant 🚀

A multimodal AI assistant that can understand and answer questions from **PDFs, URLs, and Images** using Retrieval-Augmented Generation (RAG).
Built with a focus on **accuracy, speed, and real-world usability**.

---

## ✨ Features

* 📄 **PDF Understanding** – Ask questions directly from uploaded documents
* 🌐 **URL Processing** – Extract and query information from websites
* 🖼️ **Image OCR** – Convert image text to knowledge using Tesseract
* 🧠 **Persistent Memory** – Maintains chat history across sessions
* 📚 **Source Citations** – Displays where answers are coming from
* ⚡ **Fast Responses** – Powered by Groq LLM for low-latency inference
* 🎯 **Clean UI** – Interactive Streamlit-based interface

---

## 🧠 Tech Stack

* **LangChain** – RAG pipeline and orchestration
* **Streamlit** – Frontend UI
* **Vector Database** – Embedding-based retrieval
* **Groq (LLaMA 3)** – Language model for response generation
* **Tesseract OCR** – Image text extraction

---

## ⚙️ How It Works

1. User uploads a **PDF / URL / Image**
2. Content is **loaded and chunked**
3. Chunks are converted into **vector embeddings**
4. Stored in a **vector database**
5. Relevant chunks are **retrieved using similarity search**
6. LLM generates a **context-aware answer**
7. Sources are displayed as **citations**

---

## 🚀 Future Improvements

* 🔄 Streaming responses (typing effect)
* 🤖 Multi-model fallback system
* 📊 Better chunking + retrieval optimization
* 🌍 Deployment (public web app)

---

## 💡 Key Highlights

* Designed a **multimodal RAG system**
* Implemented **cross-encoder reranking**
* Added **persistent memory using JSON**
* Built a **complete end-to-end AI application**

---

## 👨‍💻 Author

**Christian Anugrah**

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to connect!
