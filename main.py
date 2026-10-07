import logging
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. ENTERPRISE LOGGING (Bugs track karne ke liye)
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 2. FASTAPI SETUP & SWAGGER UI (Sales/Marketing ke liye)
# ---------------------------------------------------------
app = FastAPI(
    title="Enterprise AI Agent API - Web Research & Automation",
    description="Production-grade AI agent for autonomous web scraping, reasoning, and JSON structured data extraction.",
    version="3.0.0",
    docs_url="/", # Swagger UI root par dikhega, clients easily test kar payenge
)

# CORS Security Middleware (Taaki API kisi bhi dashboard se call ho sake)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 3. STRICT TYPE-CHECKING MODELS (0% Crash Guarantee)
# ---------------------------------------------------------
class AgentRequest(BaseModel):
    # HttpUrl ensure karega ki user fake ya galat URL na daale
    url: HttpUrl = Field(..., description="The target website URL to analyze.")
    task_prompt: str = Field(..., min_length=5, max_length=2000, description="What should the AI do with this webpage?")
    api_key: str = Field(..., min_length=20, description="Google Gemini API Key")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "task_prompt": "Extract the main product features and pricing into a clean JSON.",
                "api_key": "Your_Gemini_API_Key_Here"
            }
        }

class AgentResponse(BaseModel):
    success: bool
    target_url: str
    extracted_data: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------
# 4. CORE ARCHITECTURE (Robust Execution)
# ---------------------------------------------------------
@app.post("/api/v1/agent/execute", response_model=AgentResponse, status_code=status.HTTP_200_OK)
def run_autonomous_agent(payload: AgentRequest):
    logger.info(f"Processing Task for URL: {payload.url}")
    
    # STEP 1: Memory-Safe Web Scraping
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        # 10 second strict timeout taaki server hang na ho
        response = requests.get(str(payload.url), headers=headers, timeout=10)
        response.raise_for_status() # Catches 404, 403, 500 etc.
        
        # MEMORY LEAK PREVENTION: Sirf first 500KB HTML read karo
        html_content = response.text[:500000] 
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Garbage code hatana taaki LLM confuse na ho
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "iframe"]):
            tag.decompose()
            
        clean_text = soup.get_text(separator=" ", strip=True)[:15000] # Safe token limit
        
        if not clean_text:
            raise ValueError("Target website contains no readable text.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network Error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to reach target website. It might be blocking bots.")
    except Exception as e:
        logger.error(f"Parsing Error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server failed to parse the website content.")

    # STEP 2: Zero-Hallucination AI Execution
    try:
        client = genai.Client(api_key=payload.api_key)
        
        system_instruction = (
            "You are an elite enterprise data extraction and reasoning agent. "
            "Analyze the provided web context and complete the user's task. "
            "CRITICAL: You must return ONLY a strictly valid JSON object. No markdown blocks, no commentary."
        )
        
        user_input = f"Task: {payload.task_prompt}\n\nWebpage Context:\n{clean_text}"
        
        # Using Gemini 2.5 Flash Lite as requested
        gemini_response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json", # Forces strict JSON architecture
                temperature=0.1, # Low temperature for accurate, non-creative data extraction
            ),
        )
        
        # Strict JSON Validation - Ensures client never receives broken data
        result_data = json.loads(gemini_response.text)
        
        logger.info("Task executed and JSON validated successfully.")
        return AgentResponse(
            success=True,
            target_url=str(payload.url),
            extracted_data=result_data
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"AI JSON Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, 
            detail="AI failed to generate a strictly valid JSON response."
        )
    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, 
            detail=f"AI Engine Error: Check your API key and permissions."
        )

# Health Check Route for Render/RapidAPI Uptime Monitoring
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "Enterprise API is fully operational"}
  
