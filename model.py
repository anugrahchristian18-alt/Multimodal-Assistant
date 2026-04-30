from Loaders.loaders import load_pdf
from Loaders.webbase import load_url
from Loaders.img import load_image
from Processing.chunking import chunk_documents
from vectorstore.vectordb import create_vectorstore, retrieve_docs
from rag_pipeline import generate_answer
import os
import json


db = None   # NO KNOWLEDGE LOADED INITIALLY

# MEMORY FUNCTIONS (PUT HERE)


MEMORY_FILE = "chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(chat_history):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=4)
       

print("=== MULTIMODAL ASSISTANT ===")

print("Commands:")
print("/load pdf")
print("/load url")
print("/load image")
print("/exit")

# =========================
# MAIN LOOP
chat_history = load_memory()

while True:
    query = input("\nYou: ")

    if query.lower() == "exit" or query.lower() == "/exit":
        break

    # =========================
    # LOAD COMMAND HANDLING

        # ---- LOAD PDF ----
        if choice == "pdf":
            path = input("Enter PDF path: ")
            docs = load_pdf(path)

        # ---- LOAD URL ----
        elif choice == "url":
            url = input("Enter URL: ")
            docs = load_url(url)

        # ---- LOAD IMAGE ----
        elif choice == "image":
            path = input("Enter image path: ")
            docs = load_image(path)

        else:
            print("❌ Invalid type")
            continue
        
        
        
        # BUILD RAG PIPELINE
        
        # BUILD RAG PIPELINE

        chunks = chunk_documents(docs)

        if db is None:
            db = create_vectorstore(chunks)
        else:
            db.add_documents(chunks)

        print("✅ Knowledge added. Now using RAG mode.")
        continue
        
    
      # CHAT / RAG RESPONSE
      
    if db is not None:
        retrieved_docs = retrieve_docs(db, query)

        # MEMORY ADDED HERE
        answer = generate_answer(
            retrieved_docs=retrieved_docs,
            query=query,
            chat_history=chat_history
        )

        # CITATIONS ADDED HERE
        print("\n🤖 Answer:\n", answer)

        print("\n📚 Sources:")
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "N/A")

            print(f"{i}. Source: {source} | Page: {page}")

    else:
        retrieved_docs = []

        answer = generate_answer(
            retrieved_docs=[],
            query=query,
            chat_history=chat_history
            )

        print("\n🤖 Answer:\n", answer)

        # UPDATE MEMORY AFTER ANSWER
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
    save_memory(chat_history)
        
        