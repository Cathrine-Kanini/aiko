# AikoLearn Backend API

AI-Powered Multilingual Learning Assistant for Kenyan Schools following the CBC curriculum.

## Features

- 🤖 AI-powered student Q&A
- 📚 CBC-aligned curriculum content
- 👨‍🏫 Teacher lesson plan generator
- 🌐 English & Kiswahili support
- 📊 Interactive API documentation
- ✅ Comprehensive error handling

## Quick Start

### 1. Install Dependencies

\`\`\`bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

### 2. Configure Environment

\`\`\`bash
cp .env.example .env
# Edit .env and add your OpenAI API key
\`\`\`

### 3. Ingest Curriculum

\`\`\`bash
python scripts/ingest_curriculum.py
\`\`\`

### 4. Start Server

\`\`\`bash
uvicorn app.main:app --reload
\`\`\`

### 5. Test API

Open: http://localhost:8000/docs

## API Endpoints

### Student Endpoints
- `POST /api/chat` - Ask questions
- `GET /api/subjects` - List subjects

### Teacher Endpoints
- `POST /api/teacher/lesson-plan` - Generate lesson plan

### System Endpoints
- `GET /health` - Health check
- `GET /docs` - API documentation

## Testing

\`\`\`bash
pytest tests/ -v
\`\`\`

## Project Structure

\`\`\`
aikolearn-backend/
├── app/
│   ├── core/           # Configuration
│   ├── services/       # Business logic
│   ├── main.py         # FastAPI app
│   └── tests/          # Unit tests
├── scripts/            # Utility scripts
├── data/               # Curriculum data
└── logs/               # Application logs
\`\`\`

## License

MIT