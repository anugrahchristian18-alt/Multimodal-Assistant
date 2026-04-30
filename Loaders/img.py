import pytesseract
from PIL import Image
from langchain_core.documents import Document
import pytesseract
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def load_image(path):
    
    img = Image.open(path)
    
    # improve OCR quality
    img = img.convert("L")
    
    text = pytesseract.image_to_string(
        img,
        config="--psm 6"
    )
    
    return [
        Document(
            page_content=text,
            metadata={"source": path, "type": "image"}
        )
    ]