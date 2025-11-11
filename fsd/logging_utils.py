import logging
import os
from dotenv import load_dotenv

load_dotenv()

def setup_logging(log_file_path=None):
    """
    Set up logging configuration with level from environment variable.
    
    Args:
        log_file_path (str, optional): Path to log file. If None, only console logging is used.
    """
    # Get log level from environment variable, default to INFO
    log_level_str = os.getenv("LOGGING_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Common format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Handlers
    handlers = [logging.StreamHandler()]
    
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path, mode='w', encoding='utf-8'))
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True 
    )
    
    logger = logging.getLogger('logging_utils')
    logger.info(f'Logging initialized. Level: {log_level_str}')
    if log_file_path:
        logger.info(f'Log file: {log_file_path}')

def get_logger(name):
    return logging.getLogger(name)