from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Pydantic v2 config
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # Allow extra fields from environment
    )
    
    # Application
    app_name: str = "AikoLearn"
    app_version: str = "1.0.0"
    environment: str = "development"  # development, staging, production
    debug: bool = True
    
    # API
    api_v1_prefix: str = "/api/v1"
    secret_key: str = "your-secret-key-change-this-in-production"
    
    # Database
    database_url: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: str = "/tmp/aikolearn.log"  # Vercel uses /tmp for writable files
    
    # LLM Configuration
    llm_provider: str = "google"  # google, openai, anthropic, etc.
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.7
    max_tokens: int = 1000
    
    # API Keys
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Pinecone Configuration (NEW SDK)
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "aikolearn-cbc"
    pinecone_environment: str = "us-east-1"  # Not used in new SDK but kept for compatibility
    
    # ChromaDB Configuration (Fallback - NOT RECOMMENDED FOR VERCEL)
    chroma_persist_dir: str = "/tmp/chroma"  # Vercel /tmp is ephemeral!
    
    # CORS
    allowed_origins: list = ["*"]
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000


# Create single instance to be imported throughout the app
settings = Settings()