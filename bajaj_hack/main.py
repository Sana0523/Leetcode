from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any
import uvicorn
import json
import requests
import io
import time
import random
import os
from urllib.parse import urlparse
import hashlib
import logging
from contextlib import asynccontextmanager
import traceback
from datetime import datetime

# Document processing
from pypdf import PdfReader
import pdfplumber

# ML/AI components
import google.generativeai as genai

# NLP components
from nltk.corpus import stopwords
import nltk
import re

# Environment
from dotenv import load_dotenv

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables for models
gemini_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup with error handling"""
    global gemini_model
    
    try:
        logger.info("Starting model initialization...")
        
        # Download required NLTK data (optional - with error handling)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                logger.info("Downloading NLTK stopwords...")
                nltk.download('stopwords')
            except Exception as e:
                logger.warning(f"Could not download NLTK data: {e}")
        
        # Configure Gemini
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        logger.info("Configuring Gemini API...")
        genai.configure(api_key=GEMINI_API_KEY)
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        gemini_model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)
        logger.info("Gemini model configured successfully")
        
        logger.info("Model initialization completed successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't raise - allow app to start even if some models fail
        yield
    
    logger.info("Shutting down...")

app = FastAPI(
    title="HackRX Document Query API", 
    version="1.0.0",
    description="Intelligent Document Q&A System for Bajaj HackX",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Pydantic models for HackX format
class HackXRequest(BaseModel):
    documents: str = Field(..., description="URL to the document")
    questions: List[str] = Field(..., description="List of questions to answer")

class HackXResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

# Document processing classes
class DocumentProcessor:
    """Simplified document processing"""
    
    @staticmethod
    def download_document(url: str) -> bytes:
        """Download document with comprehensive error handling"""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError("Invalid URL format")
            
            # Create cache key
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_file = f"cache_{url_hash}"
            
            # Check cache
            if os.path.exists(cache_file):
                logger.info(f"Using cached document: {cache_file}")
                with open(cache_file, 'rb') as f:
                    content = f.read()
                    if len(content) > 0:
                        return content
                    else:
                        os.remove(cache_file)
            
            # Download with proper headers and timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            # Special handling for Azure Blob Storage
            if 'blob.core.windows.net' in url:
                headers.update({
                    'x-ms-version': '2020-04-08',
                    'x-ms-blob-type': 'BlockBlob'
                })
            
            logger.info(f"Downloading document from: {url}")
            
            with requests.Session() as session:
                response = session.get(
                    url, 
                    headers=headers, 
                    timeout=120,  # Increased timeout for large files
                    stream=True,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Read content with increased size limit
                content = b''
                max_size = 200 * 1024 * 1024  # 200MB
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk
                        downloaded += len(chunk)
                        if downloaded > max_size:
                            raise ValueError("Document too large (>200MB)")
                        
                        # Log progress for large files
                        if downloaded % (10 * 1024 * 1024) == 0:  # Every 10MB
                            logger.info(f"Downloaded {downloaded // 1024 // 1024}MB...")
            
            if len(content) == 0:
                raise ValueError("Downloaded document is empty")
            
            # Cache the document
            try:
                with open(cache_file, 'wb') as f:
                    f.write(content)
                logger.info(f"Document cached: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache document: {str(e)}")
            
            logger.info(f"Document downloaded successfully, size: {len(content) // 1024 // 1024}MB")
            return content
            
        except requests.RequestException as e:
            logger.error(f"Network error downloading document: {str(e)}")
            raise HTTPException(500, f"Failed to download document: {str(e)}")
        except Exception as e:
            logger.error(f"Error downloading document: {str(e)}")
            raise HTTPException(500, f"Error processing document download: {str(e)}")
    
    @staticmethod
    def extract_text_from_pdf(content: bytes) -> str:
        """Extract text from PDF with multiple fallback methods"""
        extracted_text = ""
        
        try:
            logger.info("Attempting PDF extraction with pdfplumber")
            pdf_file = io.BytesIO(content)
            
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                # Process pages in chunks to avoid memory issues
                for i, page in enumerate(pdf.pages):
                    try:
                        # Log progress for large documents
                        if i % 50 == 0:
                            logger.info(f"Processing page {i+1}/{total_pages}")
                            
                        # Extract text
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            # Clean the text
                            page_text = re.sub(r'\s+', ' ', page_text)
                            extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                            
                        # For very large documents, limit pages to avoid memory issues
                        if i > 1000:  # Limit to first 1000 pages
                            logger.info("Limiting to first 1000 pages due to size")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract page {i+1}: {str(e)}")
                        continue
                        
        except Exception as e1:
            logger.warning(f"pdfplumber failed: {str(e1)}, trying pypdf")
            
            try:
                pdf_file = io.BytesIO(content)
                pdf_reader = PdfReader(pdf_file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {total_pages} pages (pypdf)")
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        if i % 50 == 0:
                            logger.info(f"Processing page {i+1}/{total_pages} with pypdf")
                            
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            page_text = re.sub(r'\s+', ' ', page_text)
                            extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                            
                        # Limit pages for pypdf too
                        if i > 1000:
                            logger.info("Limiting to first 1000 pages due to size")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to extract page {i+1} with pypdf: {str(e)}")
                        continue
                        
            except Exception as e2:
                logger.error(f"All PDF extraction methods failed: pdfplumber: {str(e1)}, pypdf: {str(e2)}")
                raise HTTPException(500, "Unable to extract text from PDF")
        
        if not extracted_text.strip():
            raise HTTPException(400, "No text could be extracted from the PDF")
        
        logger.info(f"Successfully extracted text from PDF: {len(extracted_text)} characters")
        return extracted_text.strip()
    
    @staticmethod
    def process_document(url: str) -> str:
        """Main document processing"""
        try:
            content = DocumentProcessor.download_document(url)
            
            # Determine file type
            parsed_url = urlparse(url)
            file_path = parsed_url.path.lower()
            
            # Check file extension first
            if file_path.endswith('.pdf') or content.startswith(b'%PDF'):
                return DocumentProcessor.extract_text_from_pdf(content)
            
            # Try as plain text with different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    text = content.decode(encoding)
                    if text.strip():
                        logger.info(f"Successfully decoded as {encoding}")
                        return text
                except UnicodeDecodeError:
                    continue
            
            # Final fallback: try PDF extraction
            try:
                return DocumentProcessor.extract_text_from_pdf(content)
            except:
                raise HTTPException(400, "Unable to determine document type or extract text")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(500, f"Document processing error: {str(e)}")

class QueryProcessor:
    """Simplified query processing without sentence transformers"""
    
    @staticmethod
    def _call_gemini_with_retry(prompt: str, max_retries: int = 3) -> str:
        """Call Gemini API with retry logic"""
        if not gemini_model:
            raise Exception("Gemini model not initialized")
            
        for attempt in range(max_retries):
            try:
                response = gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=1500,
                    )
                )
                
                if response.text and response.text.strip():
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:
                    raise e
                
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
        
        raise Exception("All Gemini API attempts failed")
    
    @staticmethod
    def extract_relevant_context(question: str, document_text: str, max_context: int = 10000) -> str:
        """Extract relevant context using simple text matching"""
        # Extract key terms from question
        question_lower = question.lower()
        key_terms = []
        
        # Remove common stop words
        stop_words = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'does', 'do', 'can', 'will', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question_lower)
        key_terms = [word for word in words if word not in stop_words]
        
        if not key_terms:
            # If no key terms, return beginning of document
            return document_text[:max_context]
        
        # Find paragraphs containing key terms
        paragraphs = document_text.split('\n\n')
        relevant_paragraphs = []
        
        for para in paragraphs:
            para_lower = para.lower()
            matches = sum(1 for term in key_terms if term in para_lower)
            if matches > 0:
                relevant_paragraphs.append((para, matches))
        
        if not relevant_paragraphs:
            # No matches found, return beginning
            return document_text[:max_context]
        
        # Sort by relevance (number of matching terms)
        relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        # Combine top relevant paragraphs
        context = ""
        for para, _ in relevant_paragraphs:
            if len(context) + len(para) > max_context:
                break
            context += para + "\n\n"
        
        return context.strip() if context.strip() else document_text[:max_context]
    
    @staticmethod
    def answer_question(question: str, document_text: str) -> str:
        """Generate answer for a single question based on document text"""
        try:
            # Extract relevant context
            context = QueryProcessor.extract_relevant_context(question, document_text)
            
            # Create focused prompt
            prompt = f"""
            You are an expert document analyst. Based on the following document excerpt, answer the question accurately and concisely.

            Document excerpt:
            {context}

            Question: {question}

            Instructions:
            - Provide a clear, direct answer based on the document
            - Quote relevant parts of the text when appropriate
            - If the information is not available in the excerpt, state that clearly
            - Be precise and factual
            - Keep the answer concise but comprehensive

            Answer:
            """
            
            answer = QueryProcessor._call_gemini_with_retry(prompt)
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed for question '{question}': {str(e)}")
            return f"I encountered an error while processing this question. The document may be too complex or there may be a technical issue."

# Global storage
document_cache = {}

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - simple response for hackathon platform"""
    return {
        "message": "HackX Document Query API is running",
        "status": "active", 
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/", response_model=HackXResponse)
async def process_hackx_request(request: HackXRequest):
    """Main endpoint for HackX platform - matches the expected format"""
    start_time = time.time()
    
    try:
        logger.info(f"Received HackX request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Step 1: Process document
        logger.info("Processing document...")
        document_text = DocumentProcessor.process_document(request.documents)
        logger.info(f"Document processed successfully, length: {len(document_text):,} characters")
        
        # Step 2: Process all questions
        logger.info("Processing questions...")
        answers = []
        
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:100]}...")
            
            try:
                answer = QueryProcessor.answer_question(question, document_text)
                answers.append(answer)
                logger.info(f"Question {i+1} answered successfully")
            except Exception as e:
                logger.error(f"Failed to answer question {i+1}: {str(e)}")
                answers.append(f"Unable to process this question due to technical limitations.")
        
        processing_time = time.time() - start_time
        logger.info(f"Request completed successfully in {processing_time:.2f} seconds")
        
        return HackXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Request processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global gemini_model
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "gemini": gemini_model is not None
        },
        "memory_usage": "optimized"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )