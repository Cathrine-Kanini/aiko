from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # App Info
    app_name: str = "AikoLearn API"
    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/aikolearn.log"
    
    # Rate Limiting
    rate_limit: int = 60  # requests per minute
    
    # Model Settings
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    max_tokens: int = 1000
    
    # Vector DB
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings()

# Validate required settings
def validate_settings():
    """Validate that required settings are present"""
    errors = []
    
    if not settings.openai_api_key and not settings.anthropic_api_key:
        errors.append("Either OPENAI_API_KEY or ANTHROPIC_API_KEY must be set")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True

# Validate on import
try:
    validate_settings()
    print("✅ Configuration validated successfully")
except ValueError as e:
    print(f"⚠️  Configuration warning: {e}")