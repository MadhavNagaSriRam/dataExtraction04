'''Data Extraction API of Aadhaar card and Marksheet and TC using Google's Generative AI and FastAPI'''
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

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

def validate_pdf(filename: str, content: bytes) -> bool:
    """Validate if the file is a PDF"""
    if not filename.lower().endswith('.pdf'):
        return False
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        doc.close()
        return True
    except Exception:
        return False

def validate_image(filename: str, content: bytes) -> bool:
    """Validate if the file is an image"""
    try:
        img = PIL.Image.open(BytesIO(content))
        img.verify()  # Check if it's a valid image
        return True
    except Exception:
        return False

def convert_pdf_to_image(pdf_content: bytes) -> BytesIO:
    """Convert the first page of a PDF to an image"""
    try:
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        page = pdf_document.load_page(0)  # Load the first page
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

        # Extract text
        response = model.generate_content(["Extract all readable text from this image.", img])
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def detect_document_type(text: str) -> str:
    """Determine document type (Aadhaar or Marksheet) based on extracted text"""
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ["aadhaar", "uidai", "govt of india", "government of india"]):
        return "aadhaar"
    
    if any(keyword in text_lower for keyword in ["marksheet","roll number", "exam", "grade", "school", "university"]):
        return "marksheet"
    
    if any(keyword in text_lower for keyword in ["transfer certificate","TRANSFER CERTIFICATE", "tc number", "admission number", "conduct", "reason for leaving"]):
        return "tc"
    
    return "unknown"

def extract_data(image_source: BytesIO, prompt: str) -> dict:
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

@app.get("/extract-data/{file:path}")
async def extract_data(file: str):
    """
    Extracts data dynamically from Aadhaar or Marksheet documents (PDFs or images)
    and returns it in JSON format.
    """
    try:
        file_content = await file.read()

        # Determine file type
        is_pdf = validate_pdf(file.filename, file_content)
        is_image = validate_image(file.filename, file_content)

        if not (is_pdf or is_image):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Upload a valid PDF or image."
            )

        # Convert PDF to image if necessary
        img_data = convert_pdf_to_image(file_content) if is_pdf else BytesIO(file_content)

        if not img_data:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process {file.filename}"
            )

        # Extract raw text to determine document type
        extracted_text = extract_text_from_image(img_data)
        document_type = detect_document_type(extracted_text)

        if document_type == "aadhaar":
            prompt = """
            Analyze this Aadhaar card image and extract the following details:
            - Full name
            - Date of birth
            - Gender
            - Aadhaar number
            - Address
            -S/O, D/O
            Return the information in this JSON format:
            {
                "name": "",
                "date_of_birth": "",
                "date_of_birth_year": "",
                "gender": "",
                "aadhaar_number": "",
                "address": "",
                "Parent": "",
            }

            Guidelines:
            - Extract data exactly as printed on the card
            - Ensure Aadhaar number is 12 digits without spaces
            - Date of birth should be Null if not available day, month, year
            - Address should include all components (e.g., house number, street, city, state, PIN)

            Return only the JSON object, no additional text.
            """
        elif document_type == "marksheet" or document_type == "tc":
            prompt = """
            Analyze the given marksheet image and extract the following details in JSON format:

            Full Name: Extract the student's full name from the marksheet.
            Hall Ticket Number: Look for fields labeled as "Hall Ticket Number," "Roll Number," "Registered Number," or similar variations.
            Board of Education: Identify the course or board of education mentioned (e.g., "SSC Board," "CBSE," "ICSE," etc.).
            Religion (if available): Extract the religion of the student if explicitly mentioned.(if available)
            Total Marks : extract the total marks mentioned in the document. Or calculate the total marks based on the marks obtained in each subject.
            Identifying Marks (Moles) (if available): Look for identifying marks, moles, or physical features mentioned.(if available)
            Mother’s Name: Extract the mother’s name from the document.
            Father’s Name: Extract the father’s name from the document.
            """    
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Could not detect document type. Ensure it's an Aadhaar card or marksheet."
            )

        # Extract structured data
        extracted_data = extract_data(img_data, prompt)
        if not extracted_data:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract data from {file.filename}"
            )

        return JSONResponse(content={"document_type": document_type, "data": extracted_data})
    
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        print(f"Internal server error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
