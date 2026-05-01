from langchain_google_genai import ChatGoogleGenerativeAI
import os
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

def generate_answer(retrieved_docs, query, chat_history=None):
    if chat_history is None:
        chat_history = []

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    memory_text = ""
    for msg in chat_history[-6:]:
        memory_text += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You are a helpful and intelligient assistant.
Personality:
- Helpful, clear, and confident.
- Explain concepts simply like a friendly mentor.
- Be direct when the user is making a mistake.
- Do not overcomplicate answers.
- For coding help, explain what the code does and why it is used.
- For document questions, answer only using the provided context when possible.
- If context is missing, clearly say that the uploaded knowledge does not contain enough information.
- Always prefer practical, step-by-step guidance.

Previous conversation:
{chat_history}

Context:
{context}

User question:
{query}

If context is available, answer using it.
If context is empty, answer like a intelligient model.
"""
    
    response = llm.invoke(prompt)
    return response.content