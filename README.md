# AikoLearn API 🎓

AI-Powered Multilingual Learning Assistant for Kenyan Schools (CBC Curriculum)

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd aikolearn
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Run the application**
```bash
uvicorn main:app --reload
```

6. **Access the API**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## 🌐 Deploy to Render

### Prerequisites
- GitHub account
- Render account (free tier available)
- Google API key (get from https://aistudio.google.com/apikey)

### Deployment Steps

1. **Push to GitHub**
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

2. **Connect to Render**
- Go to https://dashboard.render.com/
- Click "New +" → "Web Service"
- Connect your GitHub repository

3. **Configure Service**
- Name: `aikolearn-api`
- Runtime: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Add Environment Variables in Render Dashboard**
```
GOOGLE_API_KEY=your_actual_api_key
ENVIRONMENT=production
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-flash
```

5. **Deploy!**
- Click "Create Web Service"
- Wait 5-10 minutes for deployment

## 📚 API Endpoints

### Student Features
- `POST /api/chat` - Ask questions
- `POST /api/quiz/generate` - Generate practice quizzes
- `POST /api/homework-help` - Get homework hints
- `POST /api/solve-problem` - Step-by-step solutions
- `GET /api/daily-tip` - Daily learning tip

### Teacher Features
- `POST /api/teacher/lesson-plan` - Generate lesson plans
- `POST /api/teacher/assessment` - Create assessments
- `POST /api/teacher/scheme-of-work` - Generate schemes of work

### Other
- `GET /health` - Health check
- `GET /api/subjects` - List subjects

## 🔧 Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `LLM_PROVIDER` | AI provider (google/openai/anthropic) | google |
| `LLM_MODEL` | Model to use | gemini-1.5-flash |
| `ENVIRONMENT` | Environment (development/production) | development |
| `PORT` | Server port (set by Render) | 8000 |

## 🐛 Troubleshooting

### Issue: Vector store not persisting
**Solution:** On Render free tier, disk storage is ephemeral. Consider:
- Using Render Disks (paid)
- Using hosted vector DB (Pinecone, Weaviate)
- Regenerating on startup

### Issue: Out of memory
**Solution:** 
- Upgrade to Starter plan (1GB RAM)
- Use API-based embeddings instead of local models

### Issue: Cold starts
**Solution:**
- Free tier spins down after 15 min inactivity
- Upgrade to paid plan for always-on service

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

For questions or support, please open an issue on GitHub.