#!/usr/bin/env python
"""
Run the FastAPI server for AI Video Analysis.

Usage:
    cd ai-core
    PYTHONPATH=src:. uv run python run_api.py [--port PORT] [--reload]

Or using uvicorn directly:
    cd ai-core
    PYTHONPATH=src:. uv run uvicorn api.main:app --reload --host 0.0.0.0

The server will be available at:
    - API Root: http://localhost:8000
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    parser = argparse.ArgumentParser(
        description="Run the AI Video Analysis API server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 for external access)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, use 1 for dev)"
    )
    
    args = parser.parse_args()
    
    from api.main import run_server
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
