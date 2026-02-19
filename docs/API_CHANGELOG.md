# API Changelog

All notable changes to the server API and ResultsContract are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Contract versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [2.0.0] — 2026-02-13

### Added — Benchmark endpoints
- `GET /api/v1/info` — Stable capabilities handshake. Returns `api_version`, `results_contract_version`, `available_backends`, `current_default_backend`, `supports_video_source`, `limits`, `assets`, `server_build`. Intended for benchmark runners and CI to verify compatibility before submitting jobs.
- `POST /api/v1/bench/run_one` — Submit one benchmark case. Supports `video_source_type=upload` (multipart file) and `video_source_type=local_asset` (video already in `DATASETS_ROOT`). Trazabilidad fields: `case_id`, `client_run_id`, `tags` (JSON dict), per-job `backend` override. Returns `job_id` + `links` to standard polling endpoints.
- `GET /api/v1/assets` — List video files available in `DATASETS_ROOT` for `local_asset` mode.

### Added — Job trazabilidad
- `VideoJob` now stores: `started_at`, `finished_at`, `source_type` (`upload`/`local_asset`), `asset_id`, `case_id`, `client_run_id`, `tags`.
- `job_meta.json` written to `data/api/results/<job_id>/job_meta.json` on every job completion or failure. Schema v1.0. Includes timestamps, source info, benchmark metadata, model config used, video metadata, and ResultsContract version.

### Fixed — `pipeline.log` now populated
- `PipelineRunner._setup_logging` now creates a unique logger per run (`PipelineRunner.<run_id>`) instead of calling `logging.basicConfig` (which is a global one-shot and caused all subsequent runs to write to the first job's log file). Logs are written to `data/api/results/<job_id>/pipeline.log`.

### Changed — `storage.py`
- `create_job()` accepts new optional parameters: `case_id`, `client_run_id`, `tags`, `source_type`, `asset_id`.
- `update_job()` accepts `started_at` and `finished_at`.
- New methods: `save_job_meta()`, `list_assets()`, `resolve_asset_path()`.

### Changed — `tasks.py`
- `process_video_task()` records `started_at` at task entry and `finished_at` at completion/failure.
- Writes `job_meta.json` on both success and failure paths.

### Added — `DATASETS_ROOT` environment variable
- Controls where `local_asset` videos are looked up. Default: `<project_root>/data/bench_assets`.
- Path-traversal protection: resolved paths are validated to be inside `DATASETS_ROOT`.

### Compatibility
- **No breaking changes** to existing mobile app endpoints (`/api/v1/videos/*`).
- `job_meta.json` is a server-side artefact and not part of any API response.
- Old `VideoJob` records without the new fields are loaded with `None` defaults (backward-compatible).

---

## [1.0.0] — 2026-01-xx (baseline before benchmark work)

### State at baseline
- `POST /api/v1/videos/upload` — upload video + disc seed (multipart)
- `GET /api/v1/videos/{id}/status` — poll job progress
- `GET /api/v1/videos/{id}/results` — download ResultsContract JSON
- `DELETE /api/v1/videos/{id}` — delete job and artefacts
- `GET /api/v1/videos` — list all jobs
- `GET/POST /api/v1/config/models` — get/set tracking backend + optional model flags
- `GET/POST /api/v1/config/tracking-backend` — get/set active backend
- `GET /health` — healthcheck

### ResultsContract v2.0.0 (defined at this baseline)
- Guaranteed: `metadata`, `tracks` (≥1 frisbee track with `mask` + `trajectory`), `metrics` (13 series), `summary`.
- Optional: person tracks, pose data.
- Smoothing: Savitzky-Golay (window=11, polyorder=3) applied to trajectory and all metric series.
- Defined in `ai-core/src/api/models.py`.

---

*Maintainer note: bump `RESULTS_CONTRACT_VERSION` in `bench.py` and `API_VERSION` in `main.py` whenever the ResultsContract or API surface changes.*
