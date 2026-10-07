import os
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. ENTERPRISE LOGGING (System monitoring ke liye)
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 2. FASTAPI SETUP (Global & Attractive Swagger UI)
# ---------------------------------------------------------
app = FastAPI(
    title="Master AI Automation & Extraction Agent API",
    description="Enterprise-grade autonomous AI agent for web scraping, reasoning, and strictly structured JSON extraction. Built with Zero-Crash Architecture.",
    version="4.0.0",
    docs_url="/", # Root path par documentation show hogi
)

# CORS Security (Kisi bhi dashboard ya frontend se call ho sakti hai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 3. STRICT TYPE-CHECKING MODELS (Data Validation)
# ---------------------------------------------------------
class AgentRequest(BaseModel):
    url: HttpUrl = Field(..., description="The target website URL to analyze (must be valid HTTP/HTTPS).")
    task_prompt: str = Field(..., min_length=5, max_length=2000, description="What should the AI do with this webpage?")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "task_prompt": "Extract the main product features and pricing into a clean JSON object."
            }
        }

class AgentResponse(BaseModel):
    success: bool
    target_url: str
    extracted_data: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------
# 4. CORE ARCHITECTURE (Robust Execution & Fallback)
# ---------------------------------------------------------
@app.post("/api/v1/agent/execute", response_model=AgentResponse, status_code=status.HTTP_200_OK)
def run_autonomous_agent(payload: AgentRequest):
    logger.info(f"Processing Task for URL: {payload.url}")
    
    # Secure API Key Fetching
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.error("CRITICAL: GEMINI_API_KEY is missing in the environment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Server Misconfiguration: API Key is missing on the server."
        )

    # --- STEP 1: Safe & Memory-Efficient Web Scraping ---
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        # 10s strict timeout prevents hanging requests
        response = requests.get(str(payload.url), headers=headers, timeout=10)
        response.raise_for_status() 
        
        # MEMORY LEAK PREVENTION: Restrict memory usage to first 500KB of HTML
        html_content = response.text[:500000] 
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Decompose junk tags to save LLM tokens and improve focus
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "iframe"]):
            tag.decompose()
            
        # Extract text and limit to 15,000 chars for safe payload
        clean_text = soup.get_text(separator=" ", strip=True)[:15000]
        
        if not clean_text:
            raise ValueError("Target website contains no readable text after cleaning.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network Error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to fetch website. Ensure the URL is accessible and not blocking bots.")
    except Exception as e:
        logger.error(f"Parsing Error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server failed to parse the website content safely.")

    # --- STEP 2: Zero-Crash AI Execution with Fallback Logic ---
    try:
        client = genai.Client(api_key=gemini_key)
        
        system_instruction = (
            "You are an elite enterprise data extraction agent. "
            "Analyze the provided web context and complete the user's task. "
            "CRITICAL INSTRUCTION: You must return ONLY a strictly valid JSON object. No markdown blocks, no commentary."
        )
        
        user_input = f"Task: {payload.task_prompt}\n\nWebpage Context:\n{clean_text}"
        
        # PRIMARY ATTEMPT: Use fast model first
        try:
            logger.info("Attempting primary model: gemini-2.5-flash")
            gemini_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
        except Exception as primary_error:
            # FALLBACK TRIGGER: If primary fails (rate limit, overload), switch silently to fallback
            logger.warning(f"Primary model failed ({str(primary_error)}). Switching to Fallback: gemini-2.5-pro")
            gemini_response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
        
        # Strict JSON Validation
        result_data = json.loads(gemini_response.text)
        
        logger.info("Task executed and JSON validated successfully.")
        return AgentResponse(
            success=True,
            target_url=str(payload.url),
            extracted_data=result_data
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"AI JSON Decoding Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, 
            detail="AI failed to generate a strictly valid JSON response."
        )
    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, 
            detail=f"AI Engine execution failed: Check quotas or API status."
        )

# ---------------------------------------------------------
# 5. SERVER HEALTH MONITORING
# ---------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "Enterprise API is fully operational, Master Node Active"}
        
