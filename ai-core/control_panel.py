#!/usr/bin/env python3
"""
AI Pipeline Control Panel

A web-based interface to control and monitor the AI video analysis pipeline.

Usage:
    cd ai-core
    PYTHONPATH=src:. uv run python control_panel.py

This will:
1. Start the control panel server on port 5000
2. Open your browser automatically
3. Provide controls for FastAPI, Ngrok, and video processing
"""

import os
import sys
import json
import signal
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from flask import Flask, render_template, request, jsonify, send_from_directory

# Configuration
CONTROL_PANEL_PORT = 5001  # 5000 is often used by AirPlay on macOS
FASTAPI_PORT = 8000
PROJECT_ROOT = Path(__file__).parent.parent
AI_CORE_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "api" / "results"

# Flask app
app = Flask(__name__, 
            template_folder=str(AI_CORE_DIR / "templates"),
            static_folder=str(AI_CORE_DIR / "static"))

# Available tracking backends
TRACKING_BACKENDS = {
    "cutie": {
        "name": "Cutie VOS",
        "description": "Visual object segmentation — tracks by appearance from initial mask",
        "requires_selection": True,
        "disc_model": "Cutie (cutie-base-mega)",
        "person_model": "YOLOv8s-seg (COCO)",
        "pose_model": "YOLOv8n-pose",
    },
    "yolo": {
        "name": "YOLO Only",
        "description": "COCO class-based detection — uses 'frisbee'/'sports ball' classes",
        "requires_selection": False,
        "disc_model": "YOLOv8s-seg (COCO 'frisbee')",
        "person_model": "YOLOv8s-seg (COCO)",
        "pose_model": "YOLOv8n-pose",
    },
}

# Global state
class AppState:
    fastapi_process: Optional[subprocess.Popen] = None
    ngrok_process: Optional[subprocess.Popen] = None
    ngrok_url: Optional[str] = None
    logs: List[Dict[str, str]] = []
    current_video: Optional[str] = None
    current_selection: Optional[Dict[str, Any]] = None
    current_job_id: Optional[str] = None
    tracking_backend: str = "cutie"  # Default to Cutie (recommended)
    enable_person_detection: bool = False  # YOLO person segmentation (optional)
    enable_pose_estimation: bool = False   # YOLO pose estimation (optional)
    
state = AppState()

def log(message: str, level: str = "info"):
    """Add a log entry."""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "message": message
    }
    state.logs.append(entry)
    # Keep only last 100 logs
    if len(state.logs) > 100:
        state.logs = state.logs[-100:]
    print(f"[{entry['time']}] [{level.upper()}] {message}")

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def get_ngrok_url() -> Optional[str]:
    """Get the public ngrok URL from its API."""
    try:
        import requests
        response = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=2)
        if response.status_code == 200:
            tunnels = response.json().get("tunnels", [])
            for tunnel in tunnels:
                if tunnel.get("proto") == "https":
                    return tunnel.get("public_url")
            if tunnels:
                return tunnels[0].get("public_url")
    except:
        pass
    return None

def check_fastapi_health() -> bool:
    """Check if FastAPI is responding."""
    try:
        import requests
        response = requests.get(f"http://localhost:{FASTAPI_PORT}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# ============================================================
# Routes - Pages
# ============================================================

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

# ============================================================
# Routes - API
# ============================================================

@app.route('/api/status')
def api_status():
    """Get current status of all services."""
    fastapi_running = check_fastapi_health()
    ngrok_running = state.ngrok_process is not None and state.ngrok_process.poll() is None
    
    if ngrok_running:
        state.ngrok_url = get_ngrok_url()
    else:
        state.ngrok_url = None
    
    # Get job statistics
    job_stats = {"pending": 0, "processing": 0, "completed_today": 0}
    jobs_file = RESULTS_DIR / "jobs.json"
    if jobs_file.exists():
        try:
            with open(jobs_file) as f:
                jobs_data = json.load(f)
            today = datetime.now().date().isoformat()
            for job in jobs_data.values():
                status = job.get("status", "")
                if status == "pending":
                    job_stats["pending"] += 1
                elif status == "processing":
                    job_stats["processing"] += 1
                elif status == "completed":
                    created = job.get("created_at", "")
                    if created.startswith(today):
                        job_stats["completed_today"] += 1
        except:
            pass
    
    backend_info = TRACKING_BACKENDS.get(state.tracking_backend, TRACKING_BACKENDS["cutie"])
    
    return jsonify({
        "fastapi": {
            "running": fastapi_running,
            "port": FASTAPI_PORT
        },
        "ngrok": {
            "running": ngrok_running,
            "url": state.ngrok_url
        },
        "current_video": state.current_video,
        "current_selection": state.current_selection,
        "current_job_id": state.current_job_id,
        "job_stats": job_stats,
        "tracking": {
            "backend": state.tracking_backend,
            "backend_info": backend_info,
            "available_backends": TRACKING_BACKENDS
        },
        "optional_models": {
            "person_detection": state.enable_person_detection,
            "pose_estimation": state.enable_pose_estimation
        }
    })

@app.route('/api/logs')
def api_logs():
    """Get recent logs."""
    return jsonify(state.logs[-50:])

@app.route('/api/videos')
def api_videos():
    """List available videos in data/raw."""
    videos = []
    if RAW_VIDEO_DIR.exists():
        for f in RAW_VIDEO_DIR.iterdir():
            if f.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                videos.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
                })
    return jsonify(sorted(videos, key=lambda x: x["name"]))

@app.route('/api/jobs')
def api_jobs():
    """List recent processing jobs."""
    jobs = []
    jobs_file = RESULTS_DIR / "jobs.json"
    if jobs_file.exists():
        try:
            with open(jobs_file) as f:
                jobs_data = json.load(f)
            for vid, job in sorted(jobs_data.items(), 
                                   key=lambda x: x[1].get('created_at', ''), 
                                   reverse=True)[:10]:
                # Extract more info from job
                # Try original_filename first, then derive from video_path
                original_filename = job.get("original_filename")
                if not original_filename:
                    # Try to extract from video_path (e.g., ".../input.mp4" or original name)
                    video_path = job.get("video_path", "")
                    if video_path:
                        # The video_id might give us info, or check if there's a better source
                        original_filename = f"video_{vid[:8]}.mp4"
                    else:
                        original_filename = "unknown.mp4"
                
                # Check for disc selection - can be "selection_data" or "disc_selection"
                disc_selection = job.get("selection_data") or job.get("disc_selection")
                has_selection = (
                    disc_selection is not None and 
                    isinstance(disc_selection, dict) and
                    disc_selection.get("center") is not None and
                    len(disc_selection.get("center", [])) >= 2
                )
                
                jobs.append({
                    "video_id": vid,
                    "status": job.get("status"),
                    "created_at": job.get("created_at"),
                    "original_filename": original_filename,
                    "has_disc_selection": has_selection,
                    "disc_selection": disc_selection if has_selection else None,
                    "tracking_backend": job.get("tracking_backend", "cutie"),
                    "message": job.get("message", "")[:50]
                })
        except Exception as e:
            print(f"[Control Panel] Error loading jobs: {e}")
    return jsonify(jobs)


@app.route('/api/jobs/<job_id>')
def api_job_detail(job_id):
    """Get detailed info about a specific job."""
    jobs_file = RESULTS_DIR / "jobs.json"
    if not jobs_file.exists():
        return jsonify({"error": "No jobs file"})
    
    try:
        with open(jobs_file) as f:
            jobs_data = json.load(f)
        
        if job_id not in jobs_data:
            return jsonify({"error": "Job not found"})
        
        job = jobs_data[job_id]
        
        # Get original filename
        original_filename = job.get("original_filename")
        if not original_filename:
            original_filename = f"video_{job_id[:8]}.mp4"
        
        # Get disc selection (can be "selection_data" or "disc_selection")
        disc_selection = job.get("selection_data") or job.get("disc_selection")
        
        # Try to get results summary if completed
        summary = None
        results_file = RESULTS_DIR / job_id / "results.json"
        if job.get("status") == "completed" and results_file.exists():
            try:
                with open(results_file) as f:
                    results = json.load(f)
                summary = results.get("summary")
            except:
                pass
        
        return jsonify({
            "video_id": job_id,
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "original_filename": original_filename,
            "disc_selection": disc_selection,
            "tracking_backend": job.get("tracking_backend", "cutie"),
            "message": job.get("message"),
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# ============================================================
# Routes - Service Control
# ============================================================

@app.route('/api/fastapi/start', methods=['POST'])
def start_fastapi():
    """Start the FastAPI server."""
    if check_fastapi_health():
        return jsonify({"success": True, "message": "FastAPI already running"})
    
    try:
        log("Starting FastAPI server...")
        env = os.environ.copy()
        env["PYTHONPATH"] = "vendors/cutie:src:."
        
        state.fastapi_process = subprocess.Popen(
            ["uv", "run", "python", "run_api.py"],
            cwd=str(AI_CORE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        # Wait for it to be ready
        for _ in range(30):
            time.sleep(0.5)
            if check_fastapi_health():
                log("FastAPI server started successfully", "success")
                # Sync all model config with FastAPI
                _sync_model_config_to_fastapi()
                backend_name = TRACKING_BACKENDS.get(state.tracking_backend, {}).get("name", state.tracking_backend)
                log(f"FastAPI synced: backend={backend_name}, person={'ON' if state.enable_person_detection else 'OFF'}, pose={'ON' if state.enable_pose_estimation else 'OFF'}")
                return jsonify({"success": True, "message": "FastAPI started"})
        
        log("FastAPI failed to start in time", "error")
        return jsonify({"success": False, "message": "Timeout waiting for FastAPI"})
        
    except Exception as e:
        log(f"Failed to start FastAPI: {e}", "error")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/fastapi/stop', methods=['POST'])
def stop_fastapi():
    """Stop the FastAPI server."""
    try:
        # Kill any process on the port
        result = subprocess.run(f"lsof -ti:{FASTAPI_PORT}", shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(f"kill -9 {pid}", shell=True)
        
        state.fastapi_process = None
        log("FastAPI server stopped", "success")
        return jsonify({"success": True, "message": "FastAPI stopped"})
        
    except Exception as e:
        log(f"Failed to stop FastAPI: {e}", "error")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/ngrok/start', methods=['POST'])
def start_ngrok():
    """Start ngrok tunnel."""
    if state.ngrok_process and state.ngrok_process.poll() is None:
        return jsonify({"success": True, "message": "Ngrok already running", "url": get_ngrok_url()})
    
    try:
        log("Starting Ngrok tunnel...")
        
        state.ngrok_process = subprocess.Popen(
            ["ngrok", "http", str(FASTAPI_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        # Wait for tunnel to establish
        for _ in range(20):
            time.sleep(0.5)
            url = get_ngrok_url()
            if url:
                state.ngrok_url = url
                log(f"Ngrok tunnel established: {url}", "success")
                return jsonify({"success": True, "message": "Ngrok started", "url": url})
        
        log("Ngrok failed to establish tunnel", "error")
        return jsonify({"success": False, "message": "Timeout waiting for Ngrok"})
        
    except Exception as e:
        log(f"Failed to start Ngrok: {e}", "error")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/ngrok/stop', methods=['POST'])
def stop_ngrok():
    """Stop ngrok tunnel."""
    try:
        subprocess.run("pkill -f ngrok", shell=True)
        state.ngrok_process = None
        state.ngrok_url = None
        log("Ngrok tunnel stopped", "success")
        return jsonify({"success": True, "message": "Ngrok stopped"})
        
    except Exception as e:
        log(f"Failed to stop Ngrok: {e}", "error")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/services/start-all', methods=['POST'])
def start_all_services():
    """Start both FastAPI and Ngrok."""
    results = {}
    
    # Start FastAPI first
    if not check_fastapi_health():
        resp = start_fastapi()
        results["fastapi"] = resp.get_json()
    else:
        results["fastapi"] = {"success": True, "message": "Already running"}
    
    # Then start Ngrok
    time.sleep(1)
    if not (state.ngrok_process and state.ngrok_process.poll() is None):
        resp = start_ngrok()
        results["ngrok"] = resp.get_json()
    else:
        results["ngrok"] = {"success": True, "message": "Already running", "url": get_ngrok_url()}
    
    return jsonify(results)

@app.route('/api/services/stop-all', methods=['POST'])
def stop_all_services():
    """Stop both FastAPI and Ngrok."""
    stop_ngrok()
    stop_fastapi()
    return jsonify({"success": True, "message": "All services stopped"})

# ============================================================
# Routes - Model Configuration
# ============================================================

@app.route('/api/tracking-backend', methods=['POST'])
def set_tracking_backend():
    """Set the tracking backend (syncs with FastAPI server if running)."""
    data = request.get_json()
    backend = data.get("backend")
    
    if backend not in TRACKING_BACKENDS:
        return jsonify({"success": False, "message": f"Unknown backend: {backend}"})
    
    state.tracking_backend = backend
    info = TRACKING_BACKENDS[backend]
    log(f"Tracking backend changed to: {info['name']}", "success")
    
    # Sync with running FastAPI server
    _sync_model_config_to_fastapi()
    
    return jsonify({"success": True, "backend": backend, "info": info})

@app.route('/api/optional-models', methods=['POST'])
def set_optional_models():
    """Toggle optional models (person detection, pose estimation)."""
    data = request.get_json()
    
    changed = []
    if "person_detection" in data:
        state.enable_person_detection = bool(data["person_detection"])
        changed.append(f"person_detection={'ON' if state.enable_person_detection else 'OFF'}")
    if "pose_estimation" in data:
        state.enable_pose_estimation = bool(data["pose_estimation"])
        changed.append(f"pose_estimation={'ON' if state.enable_pose_estimation else 'OFF'}")
    
    if changed:
        log(f"Optional models updated: {', '.join(changed)}", "success")
    
    # Sync with FastAPI server
    _sync_model_config_to_fastapi()
    
    return jsonify({
        "success": True,
        "person_detection": state.enable_person_detection,
        "pose_estimation": state.enable_pose_estimation
    })

def _sync_model_config_to_fastapi():
    """Sync all model config (backend + optional models) to FastAPI."""
    if not check_fastapi_health():
        return
    try:
        import requests as req
        req.post(
            f"http://localhost:{FASTAPI_PORT}/api/v1/config/models",
            json={
                "tracking_backend": state.tracking_backend,
                "enable_person_detection": state.enable_person_detection,
                "enable_pose_estimation": state.enable_pose_estimation,
            },
            timeout=3
        )
    except Exception as e:
        log(f"Could not sync model config with FastAPI: {e}", "warning")

# ============================================================
# Routes - Video Processing
# ============================================================

@app.route('/api/select-video', methods=['POST'])
def select_video():
    """Set the current video for processing."""
    data = request.get_json()
    video_path = data.get("path")
    
    if video_path and Path(video_path).exists():
        state.current_video = video_path
        state.current_selection = None  # Reset selection
        log(f"Selected video: {Path(video_path).name}")
        return jsonify({"success": True, "video": video_path})
    
    return jsonify({"success": False, "message": "Video not found"})

@app.route('/api/open-disc-selector', methods=['POST'])
def open_disc_selector():
    """Open the disc selection GUI tool."""
    if not state.current_video:
        return jsonify({"success": False, "message": "No video selected"})
    
    try:
        log("Opening disc selection tool...")
        selection_file = "/tmp/disc_selection_panel.json"
        
        env = os.environ.copy()
        env["PYTHONPATH"] = "src:."
        
        # Run the selector tool (positional args: video_path output_path)
        process = subprocess.Popen(
            ["uv", "run", "python", "select_disc.py", state.current_video, selection_file],
            cwd=str(AI_CORE_DIR),
            env=env
        )
        
        # Wait for it to complete
        process.wait()
        
        # Read the selection
        if Path(selection_file).exists():
            with open(selection_file) as f:
                state.current_selection = json.load(f)
            log(f"Disc selection saved: center={state.current_selection.get('center')}, radius={state.current_selection.get('radius'):.1f}", "success")
            return jsonify({"success": True, "selection": state.current_selection})
        else:
            log("No selection saved (cancelled?)", "warning")
            return jsonify({"success": False, "message": "Selection cancelled"})
            
    except Exception as e:
        log(f"Failed to open disc selector: {e}", "error")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Upload and process the selected video."""
    if not state.current_video:
        return jsonify({"success": False, "message": "No video selected"})
    
    if not check_fastapi_health():
        return jsonify({"success": False, "message": "FastAPI not running"})
    
    try:
        import requests
        
        # Determine URL to use
        base_url = f"http://localhost:{FASTAPI_PORT}"
        
        backend_info = TRACKING_BACKENDS.get(state.tracking_backend, {})
        log(f"Uploading video to {base_url} (backend: {backend_info.get('name', state.tracking_backend)})")
        
        # Prepare form data
        files = {"file": open(state.current_video, "rb")}
        data = {}
        
        if state.current_selection:
            center = state.current_selection.get("center", [])
            if len(center) >= 2:
                data["disc_center_x"] = center[0]
                data["disc_center_y"] = center[1]
            if state.current_selection.get("radius"):
                data["disc_radius"] = state.current_selection["radius"]
        
        # Send tracking backend selection
        data["tracking_backend"] = state.tracking_backend
        log(f"Disc tracking: {backend_info.get('disc_model', '?')} | Person: {backend_info.get('person_model', '?')} | Pose: {backend_info.get('pose_model', '?')}")
        
        # Upload
        response = requests.post(
            f"{base_url}/api/v1/videos/upload",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            state.current_job_id = result.get("video_id")
            log(f"Video uploaded. Job ID: {state.current_job_id}", "success")
            return jsonify({"success": True, "job_id": state.current_job_id})
        else:
            log(f"Upload failed: {response.text}", "error")
            return jsonify({"success": False, "message": response.text})
            
    except Exception as e:
        log(f"Failed to process video: {e}", "error")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/job-status')
def job_status():
    """Get the status of the current job."""
    if not state.current_job_id:
        return jsonify({"status": "none"})
    
    try:
        import requests
        response = requests.get(
            f"http://localhost:{FASTAPI_PORT}/api/v1/videos/{state.current_job_id}/status",
            timeout=5
        )
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"status": "error", "message": response.text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/job-results')
def job_results():
    """Get the results of the current job."""
    if not state.current_job_id:
        return jsonify({"error": "No job"})
    
    try:
        import requests
        response = requests.get(
            f"http://localhost:{FASTAPI_PORT}/api/v1/videos/{state.current_job_id}/results",
            timeout=10
        )
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": response.text})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/open-viewer', methods=['POST'])
def open_viewer():
    """Open the interactive analysis viewer."""
    # Get job_id from request or use current
    data = request.get_json() or {}
    job_id = data.get("job_id") or state.current_job_id
    
    if not job_id:
        return jsonify({"success": False, "message": "No job to view"})
    
    try:
        results_file = RESULTS_DIR / job_id / "results.json"
        
        if not results_file.exists():
            return jsonify({"success": False, "message": "Results not found"})
        
        log(f"Opening viewer for {job_id}...")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = "src:."
        
        subprocess.Popen(
            ["uv", "run", "python", "view_analysis.py", str(results_file)],
            cwd=str(AI_CORE_DIR),
            env=env
        )
        
        log("Viewer opened", "success")
        return jsonify({"success": True})
        
    except Exception as e:
        log(f"Failed to open viewer: {e}", "error")
        return jsonify({"success": False, "message": str(e)})

# ============================================================
# Main
# ============================================================

def open_browser():
    """Open the browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{CONTROL_PANEL_PORT}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n[Control Panel] Shutting down...")
    # Stop services
    if state.ngrok_process:
        subprocess.run("pkill -f ngrok", shell=True)
    if state.fastapi_process:
        subprocess.run(f"lsof -ti:{FASTAPI_PORT} | xargs kill -9", shell=True)
    sys.exit(0)

def kill_previous_instance():
    """Kill any previous Control Panel instance on our port."""
    try:
        result = subprocess.run(
            f"lsof -ti:{CONTROL_PANEL_PORT}",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(f"kill -9 {pid}", shell=True)
            print(f"[Control Panel] Killed previous instance on port {CONTROL_PANEL_PORT}")
            time.sleep(1)
    except Exception as e:
        pass  # Ignore errors, port might just be free


if __name__ == '__main__':
    # Kill any previous instance first
    kill_previous_instance()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║            AI Pipeline Control Panel                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Dashboard: http://localhost:{CONTROL_PANEL_PORT}                          ║
║                                                               ║
║  Press Ctrl+C to stop all services and exit                   ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    log("Control Panel starting...")
    
    # Create templates directory if needed
    templates_dir = AI_CORE_DIR / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask
    app.run(host='0.0.0.0', port=CONTROL_PANEL_PORT, debug=False, threaded=True)
