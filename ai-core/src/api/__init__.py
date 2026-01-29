"""
FastAPI Backend for AI Video Analysis.

This module provides REST API endpoints for:
- Video upload and processing
- Analysis status monitoring
- Results retrieval (tracks, metrics)
"""

from .main import app, run_server

__all__ = ["app", "run_server"]
