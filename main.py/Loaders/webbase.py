from langchain_community.document_loaders import WebBaseLoader

def load_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    # optional: clean empty docs
    cleaned_docs = [doc for doc in docs if doc.page_content.strip()]
    
    return cleaned_docs