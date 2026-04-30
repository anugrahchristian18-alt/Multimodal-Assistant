from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from PIL import Image
import pytesseract



# PDF
def load_pdf(path):
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        return docs
    except Exception as e:
        print("Error loading file")
        return []
    
    