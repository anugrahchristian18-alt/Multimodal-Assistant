from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rich import print
load_dotenv()

# 🔥 YOUR LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(retrieved_docs, query,chat_history=None):
    if chat_history is None:
        chat_history = []

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    memory_text = ""
    for msg in chat_history[-6:]:
        memory_text += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You are RAGnify, a helpful and intelligent multimodal AI assistant.

Personality:
- Helpful, clear, and confident.
- Explain concepts simply like a friendly mentor.
- Be direct when the user is making a mistake.
- Do not overcomplicate answers.
- For coding help, explain what the code does and why it is used.
- For document questions, answer using the provided context when relevant.
- Always prefer practical, step-by-step guidance.

Previous conversation:
{memory_text}

Uploaded/loaded knowledge context:
{context}

User question:
{query}

Decision rules:
- If the user is greeting, chatting casually, asking coding help, career help, general AI questions, or anything not clearly about uploaded content, answer normally.
- If the user clearly asks about the uploaded PDF/image/URL/document, use the uploaded context.
- If uploaded context is needed but not enough, say: "The uploaded knowledge does not contain enough information for this."
- Do not force uploaded context into normal conversation.
- Do not invent facts from uploaded documents.
"""
    
    response = llm.invoke(prompt)
    return response.content