from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, date
import time
import json
import re
import random
from mangum import Mangum
import asyncio  # Add this import

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
    allow_origins=["*"],
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

# ============================================
# PYDANTIC MODELS
# ============================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    grade: str
    subject: str
    language: str = "en"
    session_id: Optional[str] = None

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
    language: str = "en"

class LessonPlanResponse(BaseModel):
    lesson_plan: dict
    id: str
    created_at: datetime

class QuizRequest(BaseModel):
    topic: str
    grade: str
    subject: str
    num_questions: int = Field(default=5, ge=3, le=10)
    difficulty: str = "medium"

class HomeworkRequest(BaseModel):
    question: str = Field(..., min_length=5)
    grade: str
    subject: str
    hint_level: str = "medium"

class StepByStepRequest(BaseModel):
    problem: str = Field(..., min_length=5)
    grade: str
    subject: str

class SimilarQuestionsRequest(BaseModel):
    original_question: str
    grade: str
    subject: str
    num_similar: int = Field(default=3, ge=2, le=5)

class AssessmentRequest(BaseModel):
    subject: str
    grade: str
    topics: List[str]
    num_questions: int = Field(default=10, ge=5, le=30)
    include_marking_scheme: bool = True
    assessment_type: str = "mixed"

class SchemeOfWorkRequest(BaseModel):
    subject: str
    grade: str
    term: int = Field(..., ge=1, le=3)
    num_weeks: int = Field(default=12, ge=8, le=14)

class StudentProgress(BaseModel):
    student_name: str
    grade: str
    subject: str
    quiz_scores: List[dict] = []
    topics_covered: List[str] = []
    strengths: List[str] = []
    areas_to_improve: List[str] = []

class LearningPathRequest(BaseModel):
    current_topic: str
    grade: str
    subject: str
    mastery_level: str = "beginner"

class SimplifyRequest(BaseModel):
    text: str = Field(..., min_length=10)
    grade: str
    reading_level: str = "easy"

class VisualizeConceptRequest(BaseModel):
    concept: str
    grade: str
    subject: str

# ============================================
# BASIC ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "total_endpoints": 17
    }

@app.get("/health")
async def health_check():
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

@app.get("/api/subjects")
async def get_subjects():
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

# ============================================
# STUDENT ENDPOINTS
# ============================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Student chat - answer questions using CBC curriculum"""
    try:
        logger.info(f"Chat request - Grade {request.grade}, Subject: {request.subject}")
        
        result = await rag_service.generate_answer(
            query=request.message,
            grade=request.grade,
            subject=request.subject,
            language=request.language
        )
        
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

@app.post("/api/quiz/generate")
async def generate_quiz(request: QuizRequest):
    """Generate practice quiz questions"""
    try:
        logger.info(f"Generating quiz - Topic: {request.topic}")
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        prompt = f"""Generate {request.num_questions} multiple-choice questions for Grade {request.grade} {request.subject}.

Topic: {request.topic}
Difficulty: {request.difficulty}

Return ONLY valid JSON (no markdown):
{{
  "questions": [
    {{
      "id": 1,
      "question": "Question text?",
      "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
      "correct_answer": "A",
      "explanation": "Why this is correct"
    }}
  ]
}}

Requirements:
- CBC-aligned for Grade {request.grade}
- Use Kenyan context
- Clear questions
- Plausible options"""

        response = rag_service.llm.predict(prompt)
        cleaned = re.sub(r'```json\s*|\s*```', '', response.strip())
        
        try:
            quiz_data = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                quiz_data = json.loads(json_match.group())
            else:
                raise HTTPException(status_code=500, detail="Failed to parse quiz data")
        
        logger.info(f"✅ Quiz generated with {len(quiz_data.get('questions', []))} questions")
        
        return {
            "quiz": quiz_data,
            "topic": request.topic,
            "grade": request.grade,
            "subject": request.subject,
            "difficulty": request.difficulty,
            "total_questions": len(quiz_data.get("questions", [])),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Quiz error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/homework-help")
async def homework_help(request: HomeworkRequest):
    """Get help with homework without direct answers"""
    try:
        logger.info(f"Homework help - Grade {request.grade}")
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        hint_strategies = {
            "light": "Give ONLY a small hint to get started",
            "medium": "Guide through thinking with questions",
            "detailed": "Explain concept and show similar example"
        }
        
        prompt = f"""A Grade {request.grade} student needs help with: "{request.question}"

Subject: {request.subject}
Hint Level: {request.hint_level}

{hint_strategies[request.hint_level]}

Important: DON'T solve directly. Guide them to solve it themselves."""

        response = rag_service.llm.predict(prompt)
        
        return {
            "question": request.question,
            "hint": response,
            "hint_level": request.hint_level,
            "grade": request.grade,
            "reminder": "Try solving it yourself first, then verify with your teacher!"
        }
    except Exception as e:
        logger.error(f"Homework error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/daily-tip")
async def daily_tip(grade: str, subject: str):
    """Get daily learning tip"""
    try:
        random.seed(date.today().toordinal() + hash(f"{grade}{subject}"))
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        prompt = f"""Generate ONE short, fun learning tip for Grade {grade} {subject}.

Requirements:
- Educational but fun
- Related to real-life in Kenya
- Easy to remember
- Under 80 words
- Include an emoji

Example: "🧮 Quick trick: To multiply by 5, first multiply by 10, then divide by 2!"

Your tip:"""

        response = rag_service.llm.predict(prompt).strip()
        
        return {
            "tip": response,
            "date": date.today().isoformat(),
            "grade": grade,
            "subject": subject,
            "tip_id": f"{date.today().toordinal()}_{grade}_{subject}"
        }
    except Exception as e:
        logger.error(f"Daily tip error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/solve-problem")
async def solve_problem(request: StepByStepRequest):
    """Solve problem with step-by-step explanation"""
    try:
        logger.info(f"Step-by-step solving - Grade {request.grade}")
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        prompt = f"""Solve this step-by-step for Grade {request.grade}:

Problem: {request.problem}
Subject: {request.subject}

Provide:
1. UNDERSTAND: What are we finding?
2. PLAN: What approach to use?
3. SOLVE: Show EVERY step with explanations
4. CHECK: Verify the answer
5. SUMMARY: Final answer clearly stated

Use clear language for Grade {request.grade}."""

        response = rag_service.llm.predict(prompt)
        
        return {
            "problem": request.problem,
            "solution": response,
            "grade": request.grade,
            "subject": request.subject
        }
    except Exception as e:
        logger.error(f"Problem solving error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/similar-questions")
async def similar_questions(request: SimilarQuestionsRequest):
    """Generate similar practice questions"""
    try:
        logger.info(f"Generating {request.num_similar} similar questions")
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        prompt = f"""Based on: "{request.original_question}"

Generate {request.num_similar} similar questions for Grade {request.grade} {request.subject}.

Requirements:
- Same concept, different numbers
- Progressively slightly harder
- Include answers

Format:
Question 1: ...
Answer: ...
Working: ...

Question 2: ..."""

        response = rag_service.llm.predict(prompt)
        
        return {
            "original_question": request.original_question,
            "similar_questions": response,
            "count": request.num_similar,
            "grade": request.grade,
            "subject": request.subject
        }
    except Exception as e:
        logger.error(f"Similar questions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/explore-topic/{topic}")
async def explore_topic(topic: str, grade: str, subject: str):
    """Explore a topic with related concepts"""
    try:
        context = rag_service.retrieve_context(topic, grade, subject, k=5)
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        prompt = f"""For "{topic}" in Grade {grade} {subject}:

Provide:
1. Simple explanation (2-3 sentences)
2. Why it's important
3. Related topics
4. Real-world applications
5. Common mistakes
6. Fun fact"""

        response = rag_service.llm.predict(prompt)
        
        return {
            "topic": topic,
            "exploration": response,
            "has_curriculum_content": len(context) > 0
        }
    except Exception as e:
        logger.error(f"Explore topic error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning-path")
async def learning_path(request: LearningPathRequest):
    """Suggest next topics to learn"""
    try:
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        prompt = f"""A Grade {request.grade} student learned: {request.current_topic}

Mastery: {request.mastery_level}
Subject: {request.subject}

Suggest learning path:
1. Prerequisites to review
2. Next 3 topics (in order)
3. For each: why it's next, how it builds, time needed

Make it encouraging!"""

        response = rag_service.llm.predict(prompt)
        
        return {
            "current_topic": request.current_topic,
            "learning_path": response,
            "mastery_level": request.mastery_level
        }
    except Exception as e:
        logger.error(f"Learning path error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# TEACHER ENDPOINTS
# ============================================

@app.post("/api/teacher/lesson-plan", response_model=LessonPlanResponse)
async def generate_lesson_plan(request: LessonPlanRequest):
    """Generate CBC-aligned lesson plan"""
    try:
        logger.info(f"Lesson plan - {request.subject}, Grade {request.grade}")
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        lang = "in English" if request.language == "en" else "in Kiswahili"
        
        prompt = f"""Generate CBC-aligned lesson plan {lang}:

Subject: {request.subject}
Grade: {request.grade}
Topic: {request.topic}
Duration: {request.duration_minutes} minutes

Include:
1. LEARNING OUTCOMES (CBC)
2. KEY INQUIRY QUESTIONS
3. LEARNING EXPERIENCES
   - Introduction ({int(request.duration_minutes * 0.15)} min)
   - Main Activities ({int(request.duration_minutes * 0.60)} min)
   - Conclusion ({int(request.duration_minutes * 0.15)} min)
4. RESOURCES & MATERIALS
5. ASSESSMENT METHODS
6. DIFFERENTIATION
7. CORE COMPETENCIES & VALUES
8. REFLECTION

Format clearly for immediate use."""

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

@app.post("/api/teacher/assessment")
async def generate_assessment(request: AssessmentRequest):
    """Generate complete assessment with marking scheme"""
    try:
        logger.info(f"Assessment - {request.subject}, Grade {request.grade}")
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        topics_text = ", ".join(request.topics)
        
        type_instructions = {
            "mcq": "All multiple choice (4 options each)",
            "short_answer": "All 2-4 sentence answers",
            "essay": "All long-answer/essay type",
            "mixed": "Mix of multiple choice, short answer, essay"
        }
        
        prompt = f"""Create CBC assessment:

Subject: {request.subject}
Grade: {request.grade}
Topics: {topics_text}
Questions: {request.num_questions}
Type: {request.assessment_type}

{type_instructions[request.assessment_type]}

Format:
[HEADER]
GRADE {request.grade} {request.subject.upper()} ASSESSMENT
Topics: {topics_text}

[INSTRUCTIONS]

[QUESTIONS]
Section A: ...
1. Question (X marks)

{'[MARKING SCHEME]' if request.include_marking_scheme else ''}

Requirements:
- CBC-aligned
- Clear mark allocation
- Kenyan context"""

        response = rag_service.llm.predict(prompt)
        
        logger.info("✅ Assessment generated")
        
        return {
            "assessment": response,
            "subject": request.subject,
            "grade": request.grade,
            "topics": request.topics,
            "total_questions": request.num_questions,
            "has_marking_scheme": request.include_marking_scheme,
            "type": request.assessment_type,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/teacher/scheme-of-work")
async def generate_scheme_of_work(request: SchemeOfWorkRequest):
    """Generate term scheme of work"""
    try:
        logger.info(f"Scheme of work - {request.subject} T{request.term}")
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        prompt = f"""Generate CBC Scheme of Work:

Subject: {request.subject}
Grade: {request.grade}
Term: {request.term}
Duration: {request.num_weeks} weeks

For each week provide:
- Week number
- Strand & Sub-strand
- Learning Outcomes
- Key Questions
- Activities
- Resources
- Assessment
- Competencies
- Values
- Reflection space

Requirements:
- Proper progression
- All CBC strands
- Locally available resources
- Formative & summative assessments"""

        response = rag_service.llm.predict(prompt)
        
        logger.info("✅ Scheme of work generated")
        
        return {
            "scheme_of_work": response,
            "subject": request.subject,
            "grade": request.grade,
            "term": request.term,
            "weeks": request.num_weeks,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Scheme of work error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/teacher/progress-report")
async def generate_progress_report(progress: StudentProgress):
    """Generate student progress report"""
    try:
        avg_score = sum([s.get('score', 0) for s in progress.quiz_scores]) / len(progress.quiz_scores) if progress.quiz_scores else 0
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        prompt = f"""Generate CBC progress report:

Student: {progress.student_name}
Grade: {progress.grade}
Subject: {progress.subject}
Average: {avg_score}%

Topics: {', '.join(progress.topics_covered)}
Strengths: {', '.join(progress.strengths)}
Improve: {', '.join(progress.areas_to_improve)}

Format:
1. Overall Performance
2. Subject Progress
3. CBC Competencies
4. Recommendations
5. Next Steps

Positive and constructive."""

        response = rag_service.llm.predict(prompt)
        
        return {
            "report": response,
            "student": progress.student_name,
            "average_score": avg_score
        }
    except Exception as e:
        logger.error(f"Progress report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# ACCESSIBILITY FEATURES
# ============================================

@app.post("/api/simplify")
async def simplify_explanation(request: SimplifyRequest):
    """Simplify complex explanations"""
    try:
        logger.info(f"Simplifying for Grade {request.grade}")
        
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        levels = {
            "easy": "Very simple words, short sentences, lots of examples",
            "medium": "Clear language, some terms explained, good examples",
            "advanced": "Standard grade-level vocabulary, proper terminology"
        }
        
        prompt = f"""Rewrite for Grade {request.grade}:

ORIGINAL: {request.text}

LEVEL: {request.reading_level}
{levels[request.reading_level]}

Requirements:
- Keep SAME meaning
- Much easier to understand
- Use analogies
- Be encouraging"""

        response = rag_service.llm.predict(prompt)
        
        return {
            "original": request.text,
            "simplified": response,
            "reading_level": request.reading_level,
            "grade": request.grade
        }
    except Exception as e:
        logger.error(f"Simplification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visualize-concept")
async def visualize_concept(request: VisualizeConceptRequest):
    """Describe how to visualize/draw a concept"""
    try:
        if not rag_service.llm:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        prompt = f"""For Grade {request.grade} learning "{request.concept}" in {request.subject}:

Describe how to draw/visualize:
1. What to draw
2. Labels
3. Colors (if helpful)
4. Step-by-step
5. What each part represents

Use Kenyan classroom materials (paper, pencil, colored pencils).
Keep simple and clear."""

        response = rag_service.llm.predict(prompt)
        
        return {
            "concept": request.concept,
            "visualization_guide": response
        }
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# STARTUP/SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info("=" * 50)
    logger.info(f"📚 Vector store: {'✓' if rag_service.vectorstore else '✗'}")
    logger.info(f"🤖 LLM: {'✓' if rag_service.llm else '✗'}")
    logger.info(f"📝 Embeddings: {'✓' if rag_service.embeddings else '✗'}")
    logger.info("=" * 50)
    logger.info("✅ Application started successfully!")
    logger.info(f"📊 Total endpoints: 17")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Shutting down application...")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )

# ============================================
# ✅ MANGUM HANDLER FOR VERCEL - PUT AT THE END
# ============================================

handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)