"""
FastAPI Application for AI Video Analysis.

This is the main entry point for the REST API.
Run with: uvicorn api.main:app --reload
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import videos_router
from .storage import init_storage, get_storage
from .tasks import start_worker


# API metadata
API_TITLE = "AI Video Analysis API"
API_DESCRIPTION = """
## Weightlifting Video Analysis

This API processes weightlifting videos and extracts:

- **Object Tracking**: Disc, athlete, and barbell positions with persistent IDs
- **Physical Metrics**: Position, velocity, acceleration, energy, power
- **Summary Statistics**: Peak values, lift duration

### Workflow

1. **Upload** a video using `POST /api/v1/videos/upload`
2. **Poll status** using `GET /api/v1/videos/{video_id}/status`
3. **Get results** using `GET /api/v1/videos/{video_id}/results`
4. **Delete** (optional) using `DELETE /api/v1/videos/{video_id}`

### Response Format

Results are returned as JSON with:
- `tracks`: Per-frame detections with bounding boxes and masks
- `metrics`: Time series of physical quantities
- `summary`: Peak values and statistics

The mobile app renders overlays locally using this data.
"""
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks.
    """
    # Startup
    print("[API] Starting up...")
    
    # Initialize storage
    storage = init_storage()
    print(f"[API] Storage initialized:")
    print(f"  - Uploads: {storage.upload_dir}")
    print(f"  - Results: {storage.results_dir}")
    
    # Start background worker
    start_worker()
    print("[API] Background worker started")
    
    # Check for pending jobs
    jobs = storage.list_jobs()
    pending = [j for j in jobs.values() if j.status.value in ("pending", "processing")]
    if pending:
        print(f"[API] Found {len(pending)} pending jobs, re-queuing...")
        from .tasks import enqueue_processing
        for job in pending:
            await enqueue_processing(job.video_id)
    
    print("[API] Ready to accept requests")
    print(f"[API] Swagger UI: http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    print("[API] Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(videos_router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """API root - returns basic info."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "api": "/api/v1"
    }


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    storage = get_storage()
    jobs = storage.list_jobs()
    
    pending = sum(1 for j in jobs.values() if j.status.value == "pending")
    processing = sum(1 for j in jobs.values() if j.status.value == "processing")
    completed = sum(1 for j in jobs.values() if j.status.value == "completed")
    failed = sum(1 for j in jobs.values() if j.status.value == "failed")
    
    return {
        "status": "healthy",
        "jobs": {
            "total": len(jobs),
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed
        }
    }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to (0.0.0.0 for external access)
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn
    
    print(f"\n{'='*60}")
    print(f"  {API_TITLE} v{API_VERSION}")
    print(f"{'='*60}")
    print(f"  Server:   http://{host}:{port}")
    print(f"  Swagger:  http://localhost:{port}/docs")
    print(f"  ReDoc:    http://localhost:{port}/redoc")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(reload=True)
