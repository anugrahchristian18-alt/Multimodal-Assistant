from PIL import Image
import pytesseract
from langchain_core.documents import Document


def load_image(image_path):
    try:
        image = Image.open(image_path)

        text = pytesseract.image_to_string(image)

        if not text.strip():
            return [
                Document(
                    page_content="No readable text was found in this image.",
                    metadata={"source": image_path}
                )
            ]

        return [
            Document(
                page_content=text,
                metadata={"source": image_path}
            )
        ]

    except Exception as e:
        return [
            Document(
                page_content=f"Image OCR failed: {str(e)}",
                metadata={"source": image_path}
            )
        ]