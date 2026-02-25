"""
Robust logging configuration with structured logging and proper error handling.
Uses loguru for enhanced logging capabilities.
"""
import sys
from loguru import logger
import json
from typing import Dict, Any
from config import config

class StructuredLogger:
    """Custom logger with structured JSON output for better parsing."""
    
    def __init__(self):
        # Remove default logger
        logger.remove()
        
        # Add console handler with structured format
        logger.add(
            sys.stdout,
            format=self._structured_format,
            level=config.LOG_LEVEL,
            backtrace=True,
            diagnose=True
        )
        
        # Add file handler for persistent logs
        logger.add(
            "logs/trading_system_{time:YYYY-MM-DD}.log",
            rotation="500 MB