"""
Benchmark and capability endpoints.

Endpoints:
- GET  /api/v1/info           – Server capabilities (stable handshake for external runners)
- POST /api/v1/bench/run_one  – Submit one benchmark case (upload OR local_asset)
- GET  /api/v1/assets         – List available local-asset videos

Design decisions:
- /api/v1/info is preferred over /bench/capabilities so any client (mobile app,
  benchmark runner, CI) can discover server capabilities with a single stable URL.
- /bench namespace isolates benchmark-specific routes; normal mobile flow uses /videos.
- local_asset uses DATASETS_ROOT env var; path-traversal is prevented by resolving
  the final path and asserting it starts with the assets root.
- This module never touches the mobile /videos/* endpoints (non-breaking).
"""

import os
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..storage import get_storage
from ..tasks import enqueue_processing, get_server_model_config, get_server_tracking_backend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_CONTRACT_VERSION = "2.0.0"
API_VERSION = "2.0.0"
AVAILABLE_BACKENDS = ["cutie", "yolo"]
MAX_UPLOAD_MB = 100
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _datasets_root() -> Path:
    """Return assets root from env var, defaulting to data/bench_assets."""
    env = os.environ.get("DATASETS_ROOT", "")
    if env:
        return Path(env).expanduser().resolve()
    # Default: relative to project root
    project_root = Path(__file__).resolve().parents[4]
    return (project_root / "data" / "bench_assets").resolve()


def _git_sha() -> Optional[str]:
    """Return short HEAD SHA if inside a git repo, otherwise None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

info_router = APIRouter(tags=["info"])
bench_router = APIRouter(prefix="/bench", tags=["bench"])
assets_router = APIRouter(prefix="/assets", tags=["assets"])


# ---------------------------------------------------------------------------
# GET /api/v1/info
# ---------------------------------------------------------------------------

@info_router.get(
    "/info",
    summary="Server capabilities",
    description="""
    Stable handshake endpoint for external tools (benchmark runner, CI, mobile app).

    Returns the server's supported API version, results contract version, available
    tracking backends, current default backend, and operational limits.

    **Stability:** This endpoint is stable across patch versions. Fields may be added
    but existing fields will not be removed without a major version bump.
    """,
)
async def get_info():
    """Return server capabilities and current configuration."""
    model_config = get_server_model_config()
    current_backend = get_server_tracking_backend()
    assets_root = _datasets_root()

    return {
        "api_version": API_VERSION,
        "results_contract_version": RESULTS_CONTRACT_VERSION,
        "supports_video_source": ["upload", "local_asset"],
        "available_backends": AVAILABLE_BACKENDS,
        "current_default_backend": current_backend,
        "active_optional_models": {
            "person_detection": model_config.get("enable_person_detection", False),
            "pose_estimation": model_config.get("enable_pose_estimation", False),
        },
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "supported_formats": sorted(ALLOWED_EXTENSIONS),
            "max_duration_sec": None,  # not enforced server-side yet
            "max_concurrent_jobs": 1,  # single-worker queue
        },
        "assets": {
            "root": str(assets_root) if assets_root.exists() else None,
            "available": assets_root.exists(),
        },
        "server_build": {
            "git_sha": _git_sha(),
            "build_time": None,        # not tracked yet – future work
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "platform": platform.system(),
        },
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /api/v1/bench/run_one
# ---------------------------------------------------------------------------

@bench_router.post(
    "/run_one",
    summary="Run one benchmark case",
    description="""
    Submit a single video for analysis as part of a benchmark run.

    Supports two input modes:

    **1. Upload** (`video_source_type=upload`)
    - Provide the video file in the `file` field (multipart/form-data).
    - Same limits as `/videos/upload` (100 MB, mp4/mov/avi/mkv/webm).

    **2. Local asset** (`video_source_type=local_asset`)
    - The video is already present in the server's `DATASETS_ROOT` directory.
    - Provide `asset_id` (filename stem, e.g. `"video_test_1"`).
    - `asset_path` is accepted as an alias but is validated to be inside `DATASETS_ROOT`.

    **Trazabilidad fields (optional but recommended):**
    - `case_id`: Stable identifier of this test case in the runner's dataset.
    - `client_run_id`: ID of the overall benchmark run on the client side.
    - `tags`: JSON-encoded dict (e.g. `{"env": "indoor", "disc": "black"}`).

    **Backend override:**
    - `backend` overrides the server's current default for this job only.
      Allowed values: `cutie`, `yolo`.

    Returns the same `job_id` / status / results_url fields as `/videos/upload`,
    so existing polling logic (`/videos/{id}/status`, `/videos/{id}/results`)
    works unchanged.
    """,
)
async def bench_run_one(
    # --- video source ---
    video_source_type: str = Form(
        "upload",
        description="Input mode: 'upload' or 'local_asset'",
    ),
    file: Optional[UploadFile] = File(
        None,
        description="Video file (required when video_source_type='upload')",
    ),
    asset_id: Optional[str] = Form(
        None,
        description="Asset stem name inside DATASETS_ROOT (required when video_source_type='local_asset')",
    ),
    asset_path: Optional[str] = Form(
        None,
        description="Absolute or relative path inside DATASETS_ROOT (alias for asset_id)",
    ),
    # --- disc seed ---
    disc_center_x: Optional[float] = Form(None, description="Disc center X (pixels)"),
    disc_center_y: Optional[float] = Form(None, description="Disc center Y (pixels)"),
    disc_radius: Optional[float] = Form(None, description="Disc radius (pixels)"),
    seed_frame: int = Form(0, description="Frame index for the seed (default 0)"),
    # --- trazabilidad ---
    case_id: Optional[str] = Form(None, description="Stable test-case ID in the runner's dataset"),
    client_run_id: Optional[str] = Form(None, description="Global benchmark-run ID on the client side"),
    tags: Optional[str] = Form(None, description="JSON-encoded dict of free-form tags"),
    # --- options ---
    backend: Optional[str] = Form(None, description="Override tracking backend for this job ('cutie'/'yolo')"),
):
    """Submit one benchmark case for asynchronous processing."""
    storage = get_storage()

    # --- parse tags -------------------------------------------------------
    parsed_tags: Optional[dict] = None
    if tags:
        import json as _json
        try:
            parsed_tags = _json.loads(tags)
            if not isinstance(parsed_tags, dict):
                raise ValueError("tags must be a JSON object")
        except (ValueError, _json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid tags JSON: {exc}")

    # --- validate backend override ----------------------------------------
    if backend is not None and backend not in AVAILABLE_BACKENDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown backend '{backend}'. Allowed: {AVAILABLE_BACKENDS}",
        )

    # --- build disc selection data ----------------------------------------
    selection_data = None
    if disc_center_x is not None and disc_center_y is not None and disc_radius is not None:
        selection_data = {
            "center": [disc_center_x, disc_center_y],
            "radius": disc_radius,
            "seed_frame": seed_frame,
        }

    # =========================================================================
    # MODE 1: upload
    # =========================================================================
    if video_source_type == "upload":
        if file is None:
            raise HTTPException(
                status_code=422,
                detail="file is required when video_source_type='upload'",
            )

        # Validate extension
        if file.filename:
            ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                )

        # Read and size-check
        content = await file.read()
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(content) // (1024*1024)} MB). Maximum: {MAX_UPLOAD_MB} MB",
            )

        video_id = storage.generate_video_id()
        video_path = storage.save_uploaded_video(video_id, content, file.filename or "video.mp4")

        job = storage.create_job(
            video_id,
            video_path,
            selection_data=selection_data,
            original_filename=file.filename or "video.mp4",
            tracking_backend=backend,
            case_id=case_id,
            client_run_id=client_run_id,
            tags=parsed_tags,
            source_type="upload",
        )

    # =========================================================================
    # MODE 2: local_asset
    # =========================================================================
    elif video_source_type == "local_asset":
        # Determine lookup key: prefer asset_id, fall back to asset_path stem
        lookup = asset_id or asset_path
        if not lookup:
            raise HTTPException(
                status_code=422,
                detail="asset_id (or asset_path) is required when video_source_type='local_asset'",
            )

        assets_root = _datasets_root()
        if not assets_root.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Assets root does not exist: {assets_root}. "
                    "Set the DATASETS_ROOT environment variable to a valid directory."
                ),
            )

        resolved = storage.resolve_asset_path(lookup, assets_root)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail=f"Asset '{lookup}' not found in {assets_root}",
            )

        video_id = storage.generate_video_id()
        # For local assets we do NOT copy the file; we point the job directly at it.
        # process_video_task will create a symlink/copy as needed.
        job = storage.create_job(
            video_id,
            resolved,            # type: ignore[arg-type]  (Path is accepted)
            selection_data=selection_data,
            original_filename=resolved.name,
            tracking_backend=backend,
            case_id=case_id,
            client_run_id=client_run_id,
            tags=parsed_tags,
            source_type="local_asset",
            asset_id=lookup,
        )

    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown video_source_type '{video_source_type}'. Allowed: upload, local_asset",
        )

    # --- enqueue ----------------------------------------------------------
    await enqueue_processing(video_id)

    msg = f"Benchmark case queued. source={video_source_type}"
    if case_id:
        msg += f", case_id={case_id}"

    return {
        "job_id": video_id,
        "case_id": case_id,
        "client_run_id": client_run_id,
        "source_type": video_source_type,
        "status": "pending",
        "message": msg,
        "links": {
            "status": f"/api/v1/videos/{video_id}/status",
            "results": f"/api/v1/videos/{video_id}/results",
            "delete": f"/api/v1/videos/{video_id}",
        },
    }


# ---------------------------------------------------------------------------
# GET /api/v1/assets
# ---------------------------------------------------------------------------

@assets_router.get(
    "",
    summary="List available local assets",
    description="""
    List video files available in the server's `DATASETS_ROOT` directory.

    These can be referenced by `asset_id` in `POST /api/v1/bench/run_one` with
    `video_source_type=local_asset`.

    If `DATASETS_ROOT` does not exist or is not configured, an empty list is returned.
    """,
)
async def list_assets():
    """List videos available as local assets."""
    assets_root = _datasets_root()
    storage = get_storage()
    items = storage.list_assets(assets_root)

    return {
        "assets_root": str(assets_root),
        "root_exists": assets_root.exists(),
        "count": len(items),
        "assets": [
            {k: v for k, v in item.items() if k != "path"}  # omit server path for security
            for item in items
        ],
    }
