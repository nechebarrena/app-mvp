"""
Video processing API endpoints.

Endpoints:
- POST /upload: Upload a video for processing
- GET /{video_id}/status: Check processing status
- GET /{video_id}/results: Get analysis results
- DELETE /{video_id}: Delete video and results
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime

from ..models import (
    UploadResponse,
    StatusResponse,
    AnalysisResults,
    DeleteResponse,
    ErrorResponse,
    ProcessingStatus,
    VideoMetadata,
    MetricsSeries,
    MetricsSummary,
    Track
)
from ..storage import get_storage
from ..tasks import enqueue_processing, start_worker


router = APIRouter(prefix="/videos", tags=["videos"])


# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    summary="Upload a video for analysis",
    description="""
    Upload a video file to be processed by the AI pipeline.
    
    The video will be queued for processing and you can check the status
    using the `/status` endpoint.
    
    **Supported formats:** MP4, MOV, AVI, MKV, WebM  
    **Maximum size:** 100MB  
    **Recommended resolution:** 720p or 1080p
    """
)
async def upload_video(
    file: UploadFile = File(..., description="Video file to analyze")
):
    """Upload a video for processing."""
    storage = get_storage()
    
    # Validate file extension
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Generate video ID and save file
    video_id = storage.generate_video_id()
    video_path = storage.save_uploaded_video(
        video_id, 
        content, 
        file.filename or "video.mp4"
    )
    
    # Create job record
    job = storage.create_job(video_id, video_path)
    
    # Queue for processing
    await enqueue_processing(video_id)
    
    return UploadResponse(
        video_id=video_id,
        status=ProcessingStatus.PENDING,
        message="Video uploaded successfully. Processing will start shortly."
    )


@router.get(
    "/{video_id}/status",
    response_model=StatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Video not found"}
    },
    summary="Check processing status",
    description="""
    Check the current status of a video being processed.
    
    **Statuses:**
    - `pending`: Waiting in queue
    - `processing`: Currently being analyzed
    - `completed`: Ready to retrieve results
    - `failed`: An error occurred
    
    Poll this endpoint every 2-5 seconds while status is `pending` or `processing`.
    """
)
async def get_status(video_id: str):
    """Get processing status for a video."""
    storage = get_storage()
    job = storage.get_job(video_id)
    
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {video_id}"
        )
    
    return StatusResponse(
        video_id=job.video_id,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        message=job.message,
        created_at=job.created_at,
        updated_at=job.updated_at
    )


@router.get(
    "/{video_id}/results",
    response_model=AnalysisResults,
    responses={
        404: {"model": ErrorResponse, "description": "Video not found"},
        202: {"model": StatusResponse, "description": "Still processing"},
        500: {"model": ErrorResponse, "description": "Processing failed"}
    },
    summary="Get analysis results",
    description="""
    Retrieve the complete analysis results for a processed video.
    
    This includes:
    - **tracks**: Object detections with bounding boxes, masks, and trajectories
    - **metrics**: Time series of position, velocity, acceleration, energy, power
    - **summary**: Peak values and statistics
    
    Only available when status is `completed`.
    """
)
async def get_results(video_id: str):
    """Get analysis results for a video."""
    storage = get_storage()
    job = storage.get_job(video_id)
    
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {video_id}"
        )
    
    # Check if still processing
    if job.status == ProcessingStatus.PENDING or job.status == ProcessingStatus.PROCESSING:
        return JSONResponse(
            status_code=202,
            content={
                "video_id": job.video_id,
                "status": job.status.value,
                "progress": job.progress,
                "current_step": job.current_step,
                "message": "Video is still being processed. Please wait.",
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat()
            }
        )
    
    # Check if failed
    if job.status == ProcessingStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {job.message}"
        )
    
    # Load results
    results = storage.load_results(video_id)
    
    if results is None:
        raise HTTPException(
            status_code=500,
            detail="Results not found. Processing may have failed."
        )
    
    # Convert to response model
    try:
        return AnalysisResults(
            video_id=video_id,
            status=ProcessingStatus.COMPLETED,
            metadata=VideoMetadata(**results.get("metadata", {})),
            tracks=[Track(**t) for t in results.get("tracks", [])],
            metrics=MetricsSeries(**results.get("metrics", {})),
            summary=MetricsSummary(**results.get("summary", {})),
            processed_at=datetime.fromisoformat(results.get("processed_at", datetime.now().isoformat()))
        )
    except Exception as e:
        # Return raw results if conversion fails
        return JSONResponse(content=results)


@router.delete(
    "/{video_id}",
    response_model=DeleteResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Video not found"}
    },
    summary="Delete video and results",
    description="""
    Delete all data associated with a video, including:
    - Original uploaded video
    - Processing results
    - Job metadata
    
    This action is irreversible.
    """
)
async def delete_video(video_id: str):
    """Delete a video and all associated data."""
    storage = get_storage()
    
    deleted = storage.delete_video(video_id)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {video_id}"
        )
    
    return DeleteResponse(
        video_id=video_id,
        deleted=True,
        message="Video and all associated data have been deleted."
    )


@router.get(
    "",
    summary="List all videos",
    description="List all videos with their processing status."
)
async def list_videos():
    """List all videos."""
    storage = get_storage()
    jobs = storage.list_jobs()
    
    return {
        "count": len(jobs),
        "videos": [
            {
                "video_id": job.video_id,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat()
            }
            for job in jobs.values()
        ]
    }
