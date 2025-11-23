# app/routers/teacher.py
from fastapi import APIRouter, Depends
from app.services.llm_service import LLMService

router = APIRouter(prefix="/teacher", tags=["teacher"])

@router.post("/generate-lesson-plan")
async def generate_lesson_plan(
    subject: str,
    grade: str,
    topic: str,
    duration_minutes: int = 40,
    llm: LLMService = Depends()
):
    prompt = f"""Generate a CBC-aligned lesson plan for:
    
Subject: {subject}
Grade: {grade}
Topic: {topic}
Duration: {duration_minutes} minutes

Include:
1. Learning outcomes (from CBC)
2. Introduction (5 min)
3. Main activities (25 min)
4. Assessment (5 min)
5. Conclusion (5 min)
6. Materials needed
7. Differentiation strategies

Format as structured JSON."""
    
    response = await llm.generate(prompt)
    
    # Save to database
    lesson_plan = await db.lesson_plans.create({
        "subject": subject,
        "grade": grade,
        "topic": topic,
        "content": response
    })
    
    return lesson_plan