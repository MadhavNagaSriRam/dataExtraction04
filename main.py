from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import PIL.Image
import fitz  # PyMuPDF for PDF handling
from io import BytesIO
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API key for Google Generative AI from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is missing. Set it in the environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

# FastAPI app initialization
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

def validate_pdf(content: bytes) -> bool:
    """Validate if the content is a PDF"""
    try:
        fitz.open(stream=content, filetype="pdf").close()
        return True
    except Exception:
        return False

def validate_image(content: bytes) -> bool:
    """Validate if the content is an image"""
    try:
        img = PIL.Image.open(BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False

def convert_pdf_to_image(pdf_content: bytes) -> BytesIO:
    """Convert the first page of a PDF to an image"""
    try:
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        page = pdf_document.load_page(0)
        pix = page.get_pixmap()
        img_data = BytesIO(pix.tobytes("png"))
        pdf_document.close()
        return img_data
    except Exception as e:
        print(f"Error converting PDF to image: {str(e)}")
        return None

def extract_text_from_image(image_source: BytesIO) -> str:
    """Extract text from an image using Google's Generative AI"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        img = PIL.Image.open(image_source)
        response = model.generate_content(["Extract all readable text from this image.", img])
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def detect_document_type(text: str) -> str:
    """Determine document type (Aadhaar or Marksheet)"""
    text_lower = text.lower()
    if any(word in text_lower for word in ["aadhaar", "uidai", "govt of india", "government of india"]):
        return "aadhaar"
    if any(word in text_lower for word in ["marksheet", "roll number", "exam", "grade", "school", "university"]):
        return "marksheet"
    if any(word in text_lower for word in ["transfer certificate", "tc number", "admission number", "conduct"]):
        return "tc"
    return "unknown"

def extract_data_from_image(image_source: BytesIO, prompt: str) -> dict:
    """Extract structured data using Google's Generative AI"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        img = PIL.Image.open(image_source)
        response = model.generate_content([prompt, img])
        json_str = response.text.strip()
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"Error extracting data: {str(e)}")
        return None

@app.post("/extract-data/")
async def extract_data(file: UploadFile = File(...)):
    """
    Extracts data dynamically from Aadhaar, Marksheet, or Transfer Certificate (PDF or Image).
    Returns JSON response.
    """
    try:
        file_content = await file.read()

        is_pdf = validate_pdf(file_content)
        is_image = validate_image(file_content)

        if not (is_pdf or is_image):
            raise HTTPException(status_code=400, detail="Invalid file format. Upload a PDF or image.")

        img_data = convert_pdf_to_image(file_content) if is_pdf else BytesIO(file_content)

        if not img_data:
            raise HTTPException(status_code=400, detail="Failed to process the uploaded file.")

        extracted_text = extract_text_from_image(img_data)
        document_type = detect_document_type(extracted_text)

        if document_type == "aadhaar":
            prompt = """
            Extract the following Aadhaar details in JSON format:
            {
                "name": "",
                "date_of_birth": "",
                "gender": "",
                "aadhaar_number": "",
                "address": "",
                "parent": ""
            }
            Ensure the Aadhaar number is 12 digits. Return 'null' for missing values.
            """
        elif document_type == "marksheet" or document_type == "tc":
            prompt = """
            Extract the following marksheet details in JSON format:
            {
                "full_name": "",
                "hall_ticket_number": "",
                "board_of_education": "",
                "religion": "",
                "total_marks": "",
                "identifying_marks": "",
                "mother_name": "",
                "father_name": ""
            }
            Ensure missing values return 'null'.
            """
        else:
            raise HTTPException(status_code=400, detail="Document type not recognized. Upload Aadhaar or Marksheet.")

        extracted_data = extract_data_from_image(img_data, prompt)

        if not extracted_data:
            raise HTTPException(status_code=400, detail="Failed to extract data.")

        return JSONResponse(content={"document_type": document_type, "data": extracted_data})

    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        print(f"Internal server error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
