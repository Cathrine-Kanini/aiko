import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data

def test_chat_endpoint():
    """Test chat endpoint with valid request"""
    response = client.post("/api/chat", json={
        "message": "How do I add fractions?",
        "grade": "7",
        "subject": "math",
        "language": "en"
    })
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data
    assert len(data["response"]) > 0

def test_chat_invalid_grade():
    """Test chat with invalid grade"""
    response = client.post("/api/chat", json={
        "message": "Test",
        "grade": "10",  # Invalid
        "subject": "math",
        "language": "en"
    })
    assert response.status_code == 422  # Validation error

def test_get_subjects():
    """Test subjects endpoint"""
    response = client.get("/api/subjects")
    assert response.status_code == 200
    data = response.json()
    assert "subjects" in data
    assert len(data["subjects"]) > 0

def test_lesson_plan():
    """Test lesson plan generation"""
    response = client.post("/api/teacher/lesson-plan", json={
        "subject": "Mathematics",
        "grade": "7",
        "topic": "Fractions",
        "duration_minutes": 40,
        "language": "en"
    })
    assert response.status_code == 200
    data = response.json()
    assert "lesson_plan" in data
    assert "id" in data