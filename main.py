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

    # --- Regex Patterns (converted and fixed) ---
    company_name_pattern = r"^([\w0-9&.,\-()'\s]+)(\s(Pvt|PVT|Ltd|LLP|Inc|Corp|Co\.|Corporation|Company|Solutions?|Tooling?|Services?|Consultants?|International|Associates|Group|Limited|Enterprises|Industries|Technologies|Partners?))?$"
    name_pattern = r"([A-Z][a-zA-Z'\-\.]+(?:\s[A-Z][a-zA-Z'\-\.]+)*)"
    phone_pattern = r"(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}"
    mobile_pattern = r"(?i)(?:\b(?:cell|mobile|mob|m|M|\(M\))[\s:]*)?(\+91[\-\s]?)?[6789]\d{4}[\s]?\d{5}|\(?\d{3,4}\)?[\s.-]?\d{6,7}"
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    website_pattern = r"(?:www|http|https)(?::\/\/|\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?"
    postal_code_pattern = r"^[1-9]{1}[0-9]{2}\s{0,1}[0-9]{3}$"
    job_title_pattern = r"\b(?:Manager|Director|Engineer|Consultant|Developer|Analyst|Executive|Designer|Lead|Specialist|Officer)\b"
    linkedin_pattern = r"(?:linkedin\.com\/in\/[a-zA-Z0-9_-]+)"
    twitter_pattern = r"(?:twitter\.com\/[a-zA-Z0-9_]+)"
    facebook_pattern = r"(?:facebook\.com\/[a-zA-Z0-9._-]+)"

    # --- Field Extraction ---
    # Emails
    emails = re.findall(email_pattern, joined)

    # Phones (general + mobile)
    phones = re.findall(phone_pattern, joined)
    mobiles = re.findall(mobile_pattern, joined)

    # Flatten tuples returned by re.findall (if any)
    flat_phones = []
    for p in phones:
        if isinstance(p, tuple):
            flat_phones.append("".join(p))
        else:
            flat_phones.append(p)
    
    flat_mobiles = []
    for m in mobiles:
        if isinstance(m, tuple):
            flat_mobiles.append("".join(m))
        else:
            flat_mobiles.append(m)
    
    # Clean + deduplicate
    all_phones = list({
        re.sub(r'[\s\-]+', ' ', num).strip()
        for num in flat_phones + flat_mobiles if num.strip()
    })


    # Pincodes
    pincodes = re.findall(postal_code_pattern, joined)

    # Company
    company = None
    m = re.search(website_pattern, joined)
    if m:
        company = m.group(0).replace('www.', '')
    if not company:
        for l in lines:
            if re.search(r'\b(Service|Solutions|Corp|Company|Tree|Pvt|Ltd|LLP|Inc)\b', l, re.I):
                company = re.sub(r'[^A-Za-z0-9 &]', ' ', l).strip()
                break

    # Name
    name = None
    m = re.search(name_pattern, joined)
    if m:
        name = m.group(1)
    else:
        m = re.search(r'^\d+\s+([A-Z][a-z]+\s[A-Z][a-z]+)', text)
        if m:
            name = m.group(1)

    # Role / Designation
    role = None
    m = re.search(job_title_pattern, joined, re.I)
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

    # Social Links
    linkedin = re.findall(linkedin_pattern, joined)
    twitter = re.findall(twitter_pattern, joined)
    facebook = re.findall(facebook_pattern, joined)

    # --- Return structured data ---
    return {
        "name": name,
        "role": role,
        "company": company,
        "emails": emails,
        "phones": all_phones,
        "pincodes": pincodes,
        "address": addr,
        "linkedin": linkedin,
        "twitter": twitter,
        "facebook": facebook
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
