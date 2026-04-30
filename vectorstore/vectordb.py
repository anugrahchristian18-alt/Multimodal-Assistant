from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Create embeddinf model

embeddings = HuggingFaceEmbeddings()

def create_vectorstore(chunks):
    db = FAISS.from_documents(chunks,embeddings)
    return db
        
        
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def retrieve_docs(db, query):
    # Step 1: your existing vector search (fetch more to rerank from)
    docs = db.similarity_search(query, k=7)

    # Step 2: rerank and keep top 3
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=model, top_n=3)
    reranked = reranker.compress_documents(docs, query)

    return reranked