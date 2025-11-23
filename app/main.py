from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import time

from app.core.config import settings
from app.core.logging import logger
from app.services.rag_service import rag_service

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-Powered Multilingual Learning Assistant for Kenyan Schools (CBC)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} - {process_time:.2f}s")
    return response

# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="Student's question")
    grade: str = Field(..., pattern="^[4-8]$", description="Grade level (4-8)")
    subject: str = Field(..., description="Subject (math, science, english, kiswahili)")
    language: str = Field(default="en", pattern="^(en|sw)$", description="Response language")
    session_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "How do I add fractions with different denominators?",
                "grade": "7",
                "subject": "math",
                "language": "en"
            }
        }

class ChatResponse(BaseModel):
    response: str
    sources: List[dict]
    session_id: str
    timestamp: datetime
    has_context: bool

class LessonPlanRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    duration_minutes: int = Field(default=40, ge=20, le=120)
    language: str = Field(default="en", pattern="^(en|sw)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "subject": "Mathematics",
                "grade": "7",
                "topic": "Addition of Fractions",
                "duration_minutes": 40,
                "language": "en"
            }
        }

class LessonPlanResponse(BaseModel):
    lesson_plan: dict
    id: str
    created_at: datetime

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

# Health check
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.app_version,
        "environment": settings.environment,
        "components": {
            "vector_store": "connected" if rag_service.vectorstore else "disconnected",
            "llm": "connected" if rag_service.llm else "disconnected",
            "embeddings": "loaded" if rag_service.embeddings else "not loaded"
        }
    }

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Student chat endpoint - answer questions using CBC curriculum
    
    - **message**: The student's question
    - **grade**: Grade level (4-8)
    - **subject**: Subject area
    - **language**: Response language (en/sw)
    """
    try:
        logger.info(f"Chat request - Grade {request.grade}, Subject: {request.subject}")
        
        # Generate answer
        result = await rag_service.generate_answer(
            query=request.message,
            grade=request.grade,
            subject=request.subject,
            language=request.language
        )
        
        # Create session ID
        session_id = request.session_id or f"session_{int(time.time())}"
        
        return ChatResponse(
            response=result["answer"],
            sources=result["sources"],
            session_id=session_id,
            timestamp=datetime.now(),
            has_context=result["has_context"]
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Lesson plan generator
@app.post("/api/teacher/lesson-plan", response_model=LessonPlanResponse)
async def generate_lesson_plan(request: LessonPlanRequest):
    """
    Generate CBC-aligned lesson plan for teachers
    
    - **subject**: Subject name
    - **grade**: Grade level
    - **topic**: Lesson topic
    - **duration_minutes**: Lesson duration (20-120 min)
    - **language**: Language for lesson plan
    """
    try:
        logger.info(f"Lesson plan request - {request.subject}, Grade {request.grade}")
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        lang = "in English" if request.language == "en" else "in Kiswahili"
        
        prompt = f"""Generate a detailed, CBC-aligned lesson plan {lang} for:

Subject: {request.subject}
Grade: {request.grade}
Topic: {request.topic}
Duration: {request.duration_minutes} minutes

Structure your lesson plan with these sections:

1. LEARNING OUTCOMES (specific CBC outcomes)
   - Knowledge: What students will know
   - Skills: What students will be able to do
   - Attitudes: Values to develop

2. KEY INQUIRY QUESTIONS
   - 3-4 questions to guide learning

3. LEARNING EXPERIENCES (Activities)
   - Introduction/Set Induction ({int(request.duration_minutes * 0.15)} minutes)
   - Main Activities ({int(request.duration_minutes * 0.60)} minutes)
   - Conclusion/Closure ({int(request.duration_minutes * 0.15)} minutes)
   - Assessment ({int(request.duration_minutes * 0.10)} minutes)

4. RESOURCES & MATERIALS
   - List all required materials
   - Suggest locally available alternatives

5. ASSESSMENT METHODS
   - Formative assessment strategies
   - Success criteria

6. DIFFERENTIATION
   - Support for struggling learners
   - Extension for advanced learners

7. CORE COMPETENCIES & VALUES
   - CBC competencies addressed
   - Values integrated

8. REFLECTION
   - What went well?
   - What needs improvement?

Format as clear, structured text that a teacher can use immediately."""

        response = rag_service.llm.predict(prompt)
        
        lesson_plan = {
            "subject": request.subject,
            "grade": request.grade,
            "topic": request.topic,
            "duration_minutes": request.duration_minutes,
            "content": response,
            "language": request.language
        }
        
        lesson_id = f"lesson_{int(time.time())}"
        
        logger.info(f"✅ Lesson plan generated: {lesson_id}")
        
        return LessonPlanResponse(
            lesson_plan=lesson_plan,
            id=lesson_id,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Lesson plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# List available subjects
@app.get("/api/subjects")
async def get_subjects():
    """Get list of available subjects and grades"""
    return {
        "subjects": [
            {
                "id": "math",
                "name": "Mathematics",
                "grades": ["4", "5", "6", "7", "8"],
                "strands": ["Numbers", "Algebra", "Geometry", "Measurement", "Data"]
            },
            {
                "id": "science",
                "name": "Science",
                "grades": ["4", "5", "6", "7", "8"],
                "strands": ["Living Things", "Materials", "Energy", "Earth & Space"]
            },
            {
                "id": "english",
                "name": "English",
                "grades": ["4", "5", "6", "7", "8"],
                "strands": ["Listening & Speaking", "Reading", "Writing", "Grammar"]
            },
            {
                "id": "kiswahili",
                "name": "Kiswahili",
                "grades": ["4", "5", "6", "7", "8"],
                "strands": ["Kusoma", "Kuandika", "Sarufi", "Fasihi"]
            }
        ]
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 50)
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info("=" * 50)
    logger.info(f"📚 Vector store: {'✓' if rag_service.vectorstore else '✗'}")
    logger.info(f"🤖 LLM: {'✓' if rag_service.llm else '✗'}")
    logger.info(f"📝 Embeddings: {'✓' if rag_service.embeddings else '✗'}")
    logger.info("=" * 50)
    logger.info("✅ Application started successfully!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("👋 Shutting down application...")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )