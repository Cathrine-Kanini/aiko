import logging
import sys
from pathlib import Path
# from pythonjsonlogger import jsonlogger
from pythonjsonlogger import jsonlogger
from app.core.config import settings

def setup_logging():
    """Configure application logging"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("aikolearn")
    logger.setLevel(getattr(logging, settings.log_level))
    
    # Console handler (pretty format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler (JSON format for production)
    file_handler = logging.FileHandler(settings.log_file)
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    file_handler.setFormatter(json_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create global logger
logger = setup_logging()