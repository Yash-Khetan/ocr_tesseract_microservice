from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pytesseract
import re, json

app = FastAPI(title="OCR Business Card Extractor")
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # 👈 ensures correct path

# --- Extraction Logic ---
def extract(text: str):
    lines = [l.strip() for l in re.split(r'[\n\r|]+', text) if l.strip()]
    joined = " ".join(lines)

    # Emails
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', joined)

    # Phones
    phones = re.findall(r'\+?\d[\d\-\s]{5,}\d', joined)
    phones = list({re.sub(r'[\s\-]+', ' ', p).strip() for p in phones})

    # Pincodes
    pincodes = re.findall(r'\b\d{6}\b', joined)

    # Company
    company = None
    m = re.search(r'(?:www\.|https?://)?([A-Za-z0-9\.\-]+\.(?:com|in|co\.in|net|org))', joined)
    if m:
        company = m.group(1).replace('www.', '')
    if not company:
        for l in lines:
            if re.search(r'\b(Service|Solutions|Services|Corp|Company|Tree|Pvt|Ltd|LLP|Inc)\b', l, re.I):
                company = re.sub(r'[^A-Za-z0-9 &]', ' ', l).strip()
                break

    # Name
    name = None
    m = re.search(r'\b([A-Z][a-z]{2,}\s[A-Z][a-z]{2,})\b', joined)
    if m:
        name = m.group(1)
    else:
        m = re.search(r'^\d+\s+([A-Z][a-z]+\s[A-Z][a-z]+)', text)
        if m:
            name = m.group(1)

    # Role
    role = None
    m = re.search(r'\b(Marketing Executive|Director|Manager|CEO|CTO|Founder)\b', joined, re.I)
    if m:
        role = m.group(1)

    # Address
    addr = None
    m = re.search(r'(Plot[^=]*)', joined, re.I)
    if m:
        addr_block = m.group(1).strip()
        if pincodes:
            addr_block = re.split(r'\b' + pincodes[0] + r'\b', addr_block)[0].strip()
            addr_block = addr_block + (' ' + pincodes[0] if pincodes else '')
        addr = re.sub(r'\s{2,}', ' ', addr_block).strip()

    return {
        "name": name,
        "role": role,
        "company": company,
        "emails": emails,
        "phones": phones,
        "pincodes": pincodes,
        "address": addr,
    }


# --- OCR Endpoint ---
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        # Read image as bytes → convert to numpy array
        image_bytes = await file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Run Tesseract OCR
        custom_config = r'-l eng --oem 3 --psm 3'
        text = pytesseract.image_to_string(image, config=custom_config)

        # Extract structured info
        extracted = extract(text)
        return JSONResponse(content={"status": "success", "data": extracted, "raw_text": text})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
