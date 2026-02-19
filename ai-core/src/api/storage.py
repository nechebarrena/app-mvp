"""
Storage management for video files and processing data.

Handles:
- Video file storage
- Processing state management
- Results caching
- Cleanup policies
"""

import os
import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import threading

from .models import ProcessingStatus


# Default storage paths (relative to ai-core)
DEFAULT_UPLOAD_DIR = Path("../data/api/uploads")
DEFAULT_RESULTS_DIR = Path("../data/api/results")


@dataclass
class VideoJob:
    """Represents a video processing job."""
    video_id: str
    status: ProcessingStatus
    video_path: str
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0
    current_step: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    results_path: Optional[str] = None
    selection_data: Optional[Dict[str, Any]] = None  # Disc selection: {center: [x,y], radius: r}
    original_filename: Optional[str] = None  # Original uploaded filename
    tracking_backend: Optional[str] = None   # "cutie" or "yolo" (effective backend used)
    # Benchmark / trazabilidad
    case_id: Optional[str] = None            # Stable case ID from benchmark runner
    client_run_id: Optional[str] = None      # Global run ID from benchmark runner
    tags: Optional[Dict[str, Any]] = None    # Free-form tags {key: value}
    source_type: Optional[str] = None        # "upload" | "local_asset"
    asset_id: Optional[str] = None           # Asset ID when source_type == "local_asset"
    started_at: Optional[datetime] = None    # When processing actually started
    finished_at: Optional[datetime] = None   # When processing completed/failed
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['finished_at'] = self.finished_at.isoformat() if self.finished_at else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoJob":
        data['status'] = ProcessingStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        # Backward-compat: handle old jobs without newer fields
        for optional_field in (
            'selection_data', 'original_filename', 'tracking_backend',
            'case_id', 'client_run_id', 'tags', 'source_type', 'asset_id',
        ):
            if optional_field not in data:
                data[optional_field] = None
        for dt_field in ('started_at', 'finished_at'):
            if dt_field not in data or data[dt_field] is None:
                data[dt_field] = None
            else:
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        return cls(**data)


class StorageManager:
    """
    Manages video uploads and processing state.
    
    Thread-safe for concurrent access.
    """
    
    def __init__(
        self,
        upload_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None
    ):
        """
        Initialize storage manager.
        
        Args:
            upload_dir: Directory for uploaded videos
            results_dir: Directory for processing results
            base_dir: Base directory (defaults to ai-core/)
        """
        if base_dir is None:
            # Determine base directory from this file's location
            base_dir = Path(__file__).parent.parent.parent  # ai-core/
        
        self.base_dir = Path(base_dir)
        self.upload_dir = Path(upload_dir) if upload_dir else self.base_dir / DEFAULT_UPLOAD_DIR
        self.results_dir = Path(results_dir) if results_dir else self.base_dir / DEFAULT_RESULTS_DIR
        
        # In-memory job tracking (could be Redis/DB in production)
        self._jobs: Dict[str, VideoJob] = {}
        self._lock = threading.Lock()
        
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing jobs from disk
        self._load_jobs()
    
    def _load_jobs(self):
        """Load job state from disk on startup."""
        jobs_file = self.results_dir / "jobs.json"
        print(f"[Storage] Looking for jobs file: {jobs_file}")
        if jobs_file.exists():
            try:
                with open(jobs_file) as f:
                    jobs_data = json.load(f)
                loaded_count = 0
                for video_id, job_data in jobs_data.items():
                    try:
                        self._jobs[video_id] = VideoJob.from_dict(job_data)
                        loaded_count += 1
                    except Exception as e:
                        print(f"[Storage] Warning: Could not load job {video_id}: {e}")
                print(f"[Storage] Loaded {loaded_count} jobs from disk")
                print(f"[Storage] Job IDs: {list(self._jobs.keys())}")
            except Exception as e:
                print(f"[Storage] Warning: Could not load jobs file: {e}")
        else:
            print(f"[Storage] No jobs file found at {jobs_file}")
    
    def _save_jobs(self):
        """Persist job state to disk."""
        jobs_file = self.results_dir / "jobs.json"
        jobs_data = {vid: job.to_dict() for vid, job in self._jobs.items()}
        with open(jobs_file, 'w') as f:
            json.dump(jobs_data, f, indent=2)
    
    def generate_video_id(self) -> str:
        """Generate a unique video ID."""
        return str(uuid.uuid4())[:12]
    
    def save_uploaded_video(self, video_id: str, video_data: bytes, filename: str) -> Path:
        """
        Save an uploaded video file.
        
        Args:
            video_id: Unique video identifier
            video_data: Raw video bytes
            filename: Original filename
            
        Returns:
            Path to saved video file
        """
        # Create video-specific directory
        video_dir = self.upload_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Preserve original extension
        ext = Path(filename).suffix or ".mp4"
        video_path = video_dir / f"input{ext}"
        
        with open(video_path, 'wb') as f:
            f.write(video_data)
        
        return video_path
    
    def create_job(
        self,
        video_id: str,
        video_path: Path,
        selection_data: Optional[Dict[str, Any]] = None,
        original_filename: Optional[str] = None,
        tracking_backend: Optional[str] = None,
        case_id: Optional[str] = None,
        client_run_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = None,
        asset_id: Optional[str] = None,
    ) -> VideoJob:
        """
        Create a new processing job.
        
        Args:
            video_id: Unique video identifier
            video_path: Path to uploaded video
            selection_data: Optional disc selection {center: [x,y], radius: r}
            original_filename: Original filename of uploaded video
            
        Returns:
            Created VideoJob
        """
        now = datetime.now()
        job = VideoJob(
            video_id=video_id,
            status=ProcessingStatus.PENDING,
            video_path=str(video_path),
            created_at=now,
            updated_at=now,
            message="Video uploaded, waiting for processing",
            selection_data=selection_data,
            original_filename=original_filename,
            tracking_backend=tracking_backend,
            case_id=case_id,
            client_run_id=client_run_id,
            tags=tags,
            source_type=source_type or "upload",
            asset_id=asset_id,
        )
        
        with self._lock:
            self._jobs[video_id] = job
            self._save_jobs()
        
        return job
    
    def save_selection_data(self, video_id: str, selection_data: Dict[str, Any]) -> Path:
        """
        Save disc selection data to a JSON file.
        
        Args:
            video_id: Video identifier
            selection_data: {center: [x, y], radius: r}
            
        Returns:
            Path to the saved JSON file
        """
        upload_dir = self.upload_dir / video_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        selection_file = upload_dir / "disc_selection.json"
        with open(selection_file, 'w') as f:
            json.dump(selection_data, f, indent=2)
        
        return selection_file
    
    def get_selection_data(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get selection data for a video (from job or file)."""
        job = self.get_job(video_id)
        if job and job.selection_data:
            return job.selection_data
        
        # Try to load from file
        selection_file = self.upload_dir / video_id / "disc_selection.json"
        if selection_file.exists():
            with open(selection_file) as f:
                return json.load(f)
        
        return None
    
    def get_job(self, video_id: str) -> Optional[VideoJob]:
        """Get a job by video ID."""
        with self._lock:
            return self._jobs.get(video_id)
    
    def update_job(
        self,
        video_id: str,
        status: Optional[ProcessingStatus] = None,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        results_path: Optional[str] = None,
        tracking_backend: Optional[str] = None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
    ) -> Optional[VideoJob]:
        """
        Update a job's state.
        
        Returns the updated job or None if not found.
        """
        with self._lock:
            job = self._jobs.get(video_id)
            if job is None:
                return None
            
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if current_step is not None:
                job.current_step = current_step
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if results_path is not None:
                job.results_path = results_path
            if tracking_backend is not None:
                job.tracking_backend = tracking_backend
            if started_at is not None:
                job.started_at = started_at
            if finished_at is not None:
                job.finished_at = finished_at
            
            job.updated_at = datetime.now()
            self._save_jobs()
            
            return job
    
    def get_results_dir(self, video_id: str) -> Path:
        """Get the results directory for a video."""
        return self.results_dir / video_id
    
    def save_results(self, video_id: str, results: Dict[str, Any]) -> Path:
        """
        Save processing results to disk.
        
        Args:
            video_id: Video identifier
            results: Results dictionary
            
        Returns:
            Path to results file
        """
        results_dir = self.get_results_dir(video_id)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results_file
    
    def load_results(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load results from disk."""
        results_file = self.get_results_dir(video_id) / "results.json"
        if not results_file.exists():
            return None
        
        with open(results_file) as f:
            return json.load(f)
    
    def delete_video(self, video_id: str) -> bool:
        """
        Delete all data for a video.
        
        Returns True if deleted, False if not found.
        """
        with self._lock:
            job = self._jobs.pop(video_id, None)
            if job is None:
                return False
            
            # Delete upload directory
            upload_dir = self.upload_dir / video_id
            if upload_dir.exists():
                shutil.rmtree(upload_dir)
            
            # Delete results directory
            results_dir = self.results_dir / video_id
            if results_dir.exists():
                shutil.rmtree(results_dir)
            
            self._save_jobs()
            return True
    
    def list_jobs(self) -> Dict[str, VideoJob]:
        """List all jobs."""
        with self._lock:
            return dict(self._jobs)
    
    def save_job_meta(self, video_id: str, meta: Dict[str, Any]) -> Path:
        """
        Save job_meta.json to the results directory.
        
        job_meta.json provides full traceability: timestamps, backend used,
        input video info, benchmark metadata (case_id, client_run_id, tags).
        """
        results_dir = self.get_results_dir(video_id)
        results_dir.mkdir(parents=True, exist_ok=True)
        meta_file = results_dir / "job_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        return meta_file

    def list_assets(self, assets_root: Path) -> list:
        """
        List available local assets under assets_root.
        Returns list of dicts: {asset_id, path, size_mb}.
        """
        if not assets_root.exists():
            return []
        result = []
        for f in sorted(assets_root.iterdir()):
            if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                result.append({
                    "asset_id": f.stem,
                    "filename": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "path": str(f),
                })
        return result

    def resolve_asset_path(self, asset_id: str, assets_root: Path) -> Optional[Path]:
        """
        Resolve asset_id to an absolute path within assets_root.
        
        Security: validates the resolved path is inside assets_root to
        prevent path traversal attacks.
        """
        # Try exact stem match first (asset_id without extension)
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            candidate = (assets_root / (asset_id + ext)).resolve()
            if candidate.exists() and str(candidate).startswith(str(assets_root.resolve())):
                return candidate
        # Also accept asset_id that already includes extension
        candidate = (assets_root / asset_id).resolve()
        if candidate.exists() and str(candidate).startswith(str(assets_root.resolve())):
            return candidate
        return None

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Remove jobs older than max_age_hours.
        
        This should be called periodically (e.g., via cron or background task).
        """
        cutoff = datetime.now()
        to_delete = []
        
        with self._lock:
            for video_id, job in self._jobs.items():
                age_hours = (cutoff - job.created_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    to_delete.append(video_id)
        
        for video_id in to_delete:
            self.delete_video(video_id)
        
        return len(to_delete)


# Global storage instance
_storage: Optional[StorageManager] = None


def get_storage() -> StorageManager:
    """Get the global storage manager instance."""
    global _storage
    if _storage is None:
        _storage = StorageManager()
    return _storage


def init_storage(
    upload_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None
) -> StorageManager:
    """Initialize the global storage manager with custom paths."""
    global _storage
    _storage = StorageManager(upload_dir, results_dir, base_dir)
    return _storage
