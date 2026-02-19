# BENCHMARK_API.md — Living Documentation

> **Status:** Living document — v0.1 (February 2026)  
> **Scope:** Current FastAPI surface + benchmark feature design hooks  
> **Not yet implemented:** `/api/v1/bench/*` namespace

---

## A. Resumen ejecutivo

El servidor expone una API REST asíncrona (FastAPI) que acepta un video de levantamiento de pesas + una "seed" de selección inicial del disco, ejecuta un pipeline de visión artificial (tracking + métricas físicas) en background, y devuelve un JSON con tracks, series temporales y valores pico.

**Endpoints clave hoy:**
- `POST /api/v1/videos/upload` — sube video + seed, dispara job
- `GET  /api/v1/videos/{id}/status` — polling de progreso (0.0→1.0)
- `GET  /api/v1/videos/{id}/results` — descarga JSON de resultados
- `DELETE /api/v1/videos/{id}` — limpia todos los artefactos
- `GET/POST /api/v1/config/models` — configura modelos activos (runtime)
- `GET /health` — healthcheck con conteo de jobs

**Contrato de results (ResultsContract v2.0):** siempre incluye `metadata`, al menos un track `class_name="frisbee"` con `mask` + `trajectory`, `metrics` (13 series), y `summary`. Definido en `ai-core/src/api/models.py`.

**Qué NO hace todavía (benchmark):**
- No existe endpoint para ejecutar el pipeline sobre assets locales del servidor sin upload
- No existe endpoint de handshake/capabilities (`/info`, `/bench/capabilities`)
- No hay soporte batch (múltiples videos en un solo request)
- No hay almacenamiento de ground-truth ni comparación automática de resultados
- No hay sistema de etiquetado de jobs (tags, run_id externo, dataset_name)

Esta documentación prepara el camino para agregar un namespace `/api/v1/bench/*` que reutilice el job engine y el ResultsContract sin romper la API pública existente.

---

## B. Current API Surface

### Tabla de endpoints

| Método | Path | Descripción | Auth |
|--------|------|-------------|------|
| `POST` | `/api/v1/videos/upload` | Upload video + seed, crea job | ninguna |
| `GET` | `/api/v1/videos/{id}/status` | Estado y progreso del job | ninguna |
| `GET` | `/api/v1/videos/{id}/results` | JSON completo de resultados | ninguna |
| `DELETE` | `/api/v1/videos/{id}` | Borra video + artefactos | ninguna |
| `GET` | `/api/v1/videos` | Lista todos los jobs | ninguna |
| `GET` | `/api/v1/config/models` | Config actual de modelos | ninguna |
| `POST` | `/api/v1/config/models` | Cambia backend/optional models | ninguna |
| `GET` | `/api/v1/config/tracking-backend` | Backend activo | ninguna |
| `POST` | `/api/v1/config/tracking-backend` | Cambia backend | ninguna |
| `GET` | `/health` | Healthcheck + estadísticas | ninguna |
| `GET` | `/` | Root info | ninguna |
| `GET` | `/docs` | Swagger UI | ninguna |

---

### `POST /api/v1/videos/upload`

**Código:** `ai-core/src/api/routes/videos.py` → `async def upload_video(...)`

**Content-Type:** `multipart/form-data`

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `file` | binary | **Sí** | Video (MP4/MOV/AVI/MKV/WebM, max 100MB) |
| `disc_center_x` | float | **Sí*** | X del centro del disco en px (frame 0) |
| `disc_center_y` | float | **Sí*** | Y del centro del disco en px (frame 0) |
| `disc_radius` | float | **Sí*** | Radio del disco en px (frame 0) |
| `tracking_backend` | string | No | `"cutie"` o `"yolo"` (usa default del servidor) |

*\*Requerido para backend Cutie (default). Ignorado si no se envía; YOLO puede operar sin ellos.*

**Ejemplo curl:**
```bash
curl -X POST http://localhost:8000/api/v1/videos/upload \
  -F "file=@video.mp4" \
  -F "disc_center_x=469.5" \
  -F "disc_center_y=1438.3" \
  -F "disc_radius=127.8"
```

**Response 200:**
```json
{
  "video_id": "17fdbc3a-c5d",
  "status": "pending",
  "message": "Video uploaded successfully. Processing will start shortly. Disc selection: center=(469, 1438), radius=127"
}
```

**Errores:**
| Code | Causa |
|------|-------|
| 400 | Extensión inválida |
| 413 | Archivo > 100MB |
| 500 | Error de escritura en disco |

---

### `GET /api/v1/videos/{video_id}/status`

**Código:** `ai-core/src/api/routes/videos.py` → `async def get_status(...)`

**Response 200:**
```json
{
  "video_id": "17fdbc3a-c5d",
  "status": "processing",
  "progress": 0.43,
  "current_step": "cutie_disc_tracking",
  "message": "Tracking disc (2/5)",
  "created_at": "2026-02-14T22:35:00",
  "updated_at": "2026-02-14T22:36:20"
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `status` | enum | `pending` / `processing` / `completed` / `failed` |
| `progress` | float 0–1 | Progreso lineal (0.0→1.0) |
| `current_step` | string | Nombre técnico del step (debugging) |
| `message` | string | Texto UI-friendly para mostrar al usuario |

> **Nota importante:** La app móvil debe mostrar `message`, NO `current_step`. Ver `STEP_DISPLAY_NAMES` en `ai-core/src/api/tasks.py`.

**Progreso por etapas:**

| Rango de progress | Qué significa |
|-------------------|---------------|
| 0.00 | Encolado |
| 0.10 | Inicializando modelos |
| 0.15–0.85 | Ejecutando pipeline (lineal, 1 step = `0.70 / total_steps`) |
| 0.90 | Construyendo resultados |
| 1.00 | Completado |

**Errores:**
| Code | Causa |
|------|-------|
| 404 | `video_id` no existe |

---

### `GET /api/v1/videos/{video_id}/results`

**Código:** `ai-core/src/api/routes/videos.py` → `async def get_results(...)`

**Response 200:** Ver sección E (ResultsContract) para schema completo.

**Respuestas alternativas:**
| Code | Significado |
|------|-------------|
| 202 | Todavía procesando (mismo schema que `/status`) |
| 404 | `video_id` no existe |
| 500 | Falló el procesamiento o resultados no encontrados |

---

### `DELETE /api/v1/videos/{video_id}`

**Código:** `ai-core/src/api/routes/videos.py` → `async def delete_video(...)`

Borra `data/api/uploads/{id}/` + `data/api/results/{id}/` + entrada en `jobs.json`.

**Response 200:**
```json
{
  "video_id": "17fdbc3a-c5d",
  "deleted": true,
  "message": "Video and all associated data have been deleted."
}
```

---

### `GET/POST /api/v1/config/models`

**Código:** `ai-core/src/api/main.py` → `get_models_config()` / `set_models_config()`  
**Backend functions:** `ai-core/src/api/tasks.py` → `get_server_model_config()` / `set_server_model_config()`

Config almacenada en memoria en `_server_config` dict (se resetea al reiniciar el servidor).

**GET response:**
```json
{
  "tracking_backend": "cutie",
  "enable_person_detection": false,
  "enable_pose_estimation": false
}
```

**POST body:**
```json
{
  "tracking_backend": "cutie",
  "enable_person_detection": false,
  "enable_pose_estimation": false
}
```

---

### `GET /health`

**Código:** `ai-core/src/api/main.py` → `health_check()`

```json
{
  "status": "healthy",
  "jobs": {
    "total": 12,
    "pending": 0,
    "processing": 1,
    "completed": 10,
    "failed": 1
  }
}
```

---

## C. Job Execution Model

### Mecanismo de ejecución

**Asíncrono con worker único** basado en `asyncio.Queue`.

```
upload_video()
   └─► enqueue_processing(video_id)
            └─► _task_queue.put(video_id)
                     └─► _worker() loop
                              └─► process_video_task(video_id)   ← asyncio.to_thread()
                                       └─► PipelineRunner.run()  ← bloqueante en thread
```

**Archivos clave:**
- `ai-core/src/api/tasks.py` — `enqueue_processing()`, `_worker()`, `start_worker()`, `process_video_task()`
- `ai-core/src/pipeline/runner.py` — `PipelineRunner.run()` (ejecuta pasos en secuencia)
- `ai-core/src/api/main.py` (lifespan) — llama `start_worker()` al arrancar; re-encola jobs `pending`/`processing` encontrados en disco

> **Concurrencia:** el worker es single-threaded (`asyncio.Queue` + un único `create_task`). Solo se procesa un video a la vez.

### Secuencia completa de `process_video_task()`

```
1. update_job(status=PROCESSING, progress=0.0, message="Starting analysis...")
2. Crea symlink/copia en data/raw/{video_id}.mp4
3. create_api_pipeline_config() → config dict
4. yaml.dump(config) → data/api/results/{id}/pipeline_config.yaml
5. update_job(progress=0.1, message="Initializing models...")
6. PipelineRunner(config_yaml_path)  ← carga YAML, registra módulos
7. runner.on_step_start = on_step_progress  ← callback de progreso
8. update_job(progress=0.15, message="Processing video...")
9. await asyncio.to_thread(runner.run)  ← BLOQUEA hasta completar
10. update_job(progress=0.9, message="Building results...")
11. build_api_results() → dict
12. storage.save_results() → data/api/results/{id}/results.json
13. update_job(status=COMPLETED, progress=1.0, message="Analysis complete")
```

Definido en: `ai-core/src/api/tasks.py` → `async def process_video_task(video_id: str)`

### Almacenamiento en disco

```
data/
├── api/
│   ├── uploads/
│   │   └── {video_id}/
│   │       ├── input.mp4               ← video original
│   │       └── disc_selection.json     ← {center:[x,y], radius:r}
│   └── results/
│       ├── jobs.json                   ← estado persistido de todos los jobs
│       └── {video_id}/
│           ├── pipeline_config.yaml    ← config YAML exacta usada
│           ├── pipeline.log            ← log del runner (actualmente vacío — ver TODO)
│           ├── cutie_disc_tracking_output.json  ← artefacto intermedio (si save=true)
│           ├── metrics_calculator_output.json   ← métricas completas (row-per-frame)
│           ├── metrics_calculator_output.csv    ← mismas métricas en CSV
│           ├── metrics_calculator_output_summary.json
│           └── results.json            ← RESPUESTA FINAL de /results
data/raw/
│   └── {video_id}.mp4                  ← symlink → uploads/{id}/input.mp4
```

**`jobs.json`** — persiste el estado de todos los `VideoJob` entre reinicios del servidor. Cargado en `StorageManager._load_jobs()` → `ai-core/src/api/storage.py`.

### Actualización de progreso

La función `on_step_progress(step_name, step_idx, total_steps)` se registra como callback en `PipelineRunner.on_step_start`. Cada vez que el runner inicia un step, llama al callback, que:
1. Calcula `progress = 0.15 + (step_idx / total_steps) * 0.70`
2. Convierte `step_name` a texto legible via `STEP_DISPLAY_NAMES` dict
3. Llama `storage.update_job(...)` → persiste en `jobs.json`

Código: `ai-core/src/api/tasks.py` → `on_step_progress` (closure dentro de `process_video_task`)

---

## D. Input "Seed" / Initialization

### Representación

La seed es un **círculo en el primer frame**: centro `(cx, cy)` en píxeles + radio `r` en píxeles.

```json
{
  "center": [469.49036, 1438.331],
  "radius": 127.7585
}
```

El sistema de coordenadas es imagen estándar: `(0,0)` en esquina superior izquierda, Y crece hacia abajo.

### Flujo de la seed

```
POST /upload
  disc_center_x, disc_center_y, disc_radius
        │
        ▼
upload_video() [routes/videos.py]
  selection_data = {"center": [cx, cy], "radius": r}
  storage.save_selection_data(video_id, selection_data)  → uploads/{id}/disc_selection.json
  storage.create_job(..., selection_data=selection_data)  → jobs.json
        │
        ▼
process_video_task() [tasks.py]
  selection_data = job.selection_data
        │
        ▼
create_api_pipeline_config(..., selection_data=selection_data)
        │
        ├─► [Cutie] step "cutie_disc_tracking".params.initial_selection
        │     = {center: [cx, cy], radius: r}
        │     CutieTracker._get_selection() lee esta key
        │     → _create_circular_mask(h, w, center, radius) → tensor de máscara inicial
        │     → processor.step(frame, initial_mask_tensor, objects=[1])  ← solo frame 0
        │
        └─► [Cutie] step "disc_tracking".params.initial_selection
              = {class_name: "frisbee", center: [cx, cy], radius: r}
              ModelTracker usa esto para inicializar la bounding region
```

**Archivos de consumo:**
- `ai-core/src/perception/cutie_tracker.py` → `_get_selection()`, `_create_circular_mask()`
- `ai-core/src/analysis/model_tracker.py` → busca `initial_selection` en params (TODO: verificar exactamente cómo lo usa)
- `ai-core/src/api/tasks.py` → `_create_cutie_pipeline_config()` (líneas ~147–190)

### Impacto por backend

| Backend | Seed obligatoria | Uso |
|---------|-----------------|-----|
| `cutie` | **Sí** | Genera máscara circular inicial; sin ella no hay tracking |
| `yolo` | No | Opcional; puede usarse para inicializar bounding region del tracker |

### Scale (px → metros)

La seed también se usa para calcular la escala pixel/metro:

```python
# ai-core/src/analysis/metrics_calculator.py
# Si disc_diameter_m = 0.45m y radius_px = 127.76px:
scale_m_per_px = disc_diameter_m / (2 * radius_px)
```

La seed **no se re-envía en cada request de la app móvil**; se captura una vez en la pantalla de selección del disco antes del upload.

---

## E. ResultsContract

**Versión del contrato:** `2.0.0` (documentado en `docs/api_guide.md`)  
**Definición Pydantic:** `ai-core/src/api/models.py`  
**Función de construcción:** `ai-core/src/api/tasks.py` → `build_api_results()`  
**Archivo en disco:** `data/api/results/{video_id}/results.json`

### Schema completo

```json
{
  "video_id": "17fdbc3a-c5d",
  "status": "completed",
  "processed_at": "2026-02-14T22:39:48.005199",
  
  "metadata": {
    "fps": 29.475,
    "width": 1080,
    "height": 1920,
    "duration_s": 3.90,
    "total_frames": 115
  },
  
  "tracks": [
    {
      "track_id": 1,
      "class_name": "frisbee",
      "trajectory": [
        [472.12, 1434.73],
        [471.12, 1434.25]
      ],
      "frames": {
        "0": {
          "confidence": 0.995,
          "bbox": {"x1": 342.0, "y1": 1310.0, "x2": 595.0, "y2": 1566.0},
          "mask": [[455, 1310], [453, 1312], [442, 1312], "..."]
        }
      }
    }
  ],
  
  "metrics": {
    "frames":             [0, 1, 2, "..."],
    "time_s":             [0.0, 0.034, 0.068, "..."],
    "x_m":                [0.469, 0.470, "..."],
    "y_m":                [1.437, 1.437, "..."],
    "height_m":           [0.0, 0.0001, "..."],
    "vx_m_s":             [0.003, 0.004, "..."],
    "vy_m_s":             [-0.004, -0.005, "..."],
    "speed_m_s":          [0.004, 0.006, "..."],
    "accel_m_s2":         [0.046, 0.023, "..."],
    "kinetic_energy_j":   [0.001, 0.002, "..."],
    "potential_energy_j": [0.0, 0.12, "..."],
    "total_energy_j":     [0.001, 0.122, "..."],
    "power_w":            [0.018, 0.034, "..."]
  },
  
  "summary": {
    "peak_speed_m_s":  1.9415,
    "peak_power_w":    1471.29,
    "max_height_m":    0.7887,
    "min_height_m":    0.0,
    "lift_duration_s": 3.90,
    "total_frames":    115
  }
}
```

### Reglas de contrato

| Campo | Garantizado | Condición |
|-------|------------|-----------|
| `metadata` | ✅ Siempre | — |
| Track `class_name="frisbee"` | ✅ Siempre | — |
| `mask` en frames del disco | ✅ Siempre | Backend Cutie |
| `bbox` en frames | ⚠️ Opcional | Puede ser `null` para disco |
| `trajectory` (array [[x,y]]) | ✅ Siempre | Suavizado con Savitzky-Golay window=11 |
| `metrics` (13 series) | ✅ Siempre | Puede tener arrays vacíos si pipeline falla |
| `summary` | ✅ Siempre | Valores 0.0 si sin métricas |
| Track `class_name="person"` | ⚠️ Opcional | Solo si `enable_person_detection=true` |

### Suavizado aplicado

La trayectoria y las métricas pasan por Savitzky-Golay antes de ser escritas:
1. `build_api_results()` → trayectoria smoothed con `savgol_filter(window=11, polyorder=3)`
2. `track_refiner` → posiciones bbox/mask centroid smoothed con `savgol(window=11)` antes de pasar a metrics
3. `metrics_calculator._smooth()` → velocidades smoothed con `savgol(window=11, polyorder=3)` antes de derivar aceleración

### Compatibilidad con app móvil

| Feature mobile | Campos requeridos |
|---------------|-------------------|
| Overlay contorno disco | `tracks[?].frames[N].mask` |
| Trayectoria disco | `tracks[?].trajectory` |
| Gráfico velocidad/potencia | `metrics.speed_m_s`, `metrics.power_w` |
| Gráfico altura | `metrics.height_m`, `metrics.time_s` |
| Pantalla de resultados | `summary.*` |
| Sincronización con video | `metadata.fps` + índice de frame |

---

## F. Observability & Debug Artifacts

### Logs

| Artefacto | Path | Estado |
|-----------|------|--------|
| Pipeline log | `data/api/results/{id}/pipeline.log` | ⚠️ Vacío en jobs API (el logger escribe a consola, no al archivo del job) |
| Consola del servidor | stdout del proceso FastAPI | ✅ Activo — muestra `[Task]`, `[Config]`, `[TrackRefiner]`, etc. |

> **TODO:** El `_setup_logging()` de `PipelineRunner` (`ai-core/src/pipeline/runner.py` línea 117) configura un `FileHandler` hacia `pipeline.log`, pero ese path apunta al `output_dir` del runner (que en API va a `data/outputs/...`, no a `data/api/results/{id}/`). Por eso el log queda vacío en el artefacto del job. Para debug real, revisar la consola del proceso.

### Artefactos intermedios

| Archivo | Cuándo existe | Qué contiene |
|---------|---------------|--------------|
| `pipeline_config.yaml` | Siempre | Config YAML exacta pasada al runner |
| `cutie_disc_tracking_output.json` | `save_output: true` en step | `Dict[frame_idx, List[Detection]]` del tracker |
| `metrics_calculator_output.json` | Siempre | Array de dicts, un row por frame |
| `metrics_calculator_output.csv` | Siempre | Mismas métricas en CSV (cols: frame_idx, time_s, x_px, y_px, x_m, ...) |
| `metrics_calculator_output_summary.json` | Siempre | Estadísticas de posición, velocidad, energía, potencia |
| `results.json` | Solo si completado | Payload final de `/results` |

### Modos Memory vs Disk

El runner soporta dos modos de paso a paso (definido en `ai-core/src/pipeline/runner.py`):

| Modo | `save_output` en step | Descripción |
|------|----------------------|-------------|
| **Memory** (default API) | `false` | Steps pasan datos en RAM via `runner.step_outputs` dict |
| **Disk** (debug) | `true` en step | Step serializa su output a `{step_name}_output.json` en disco |

En el pipeline API por defecto, solo `cutie_disc_tracking` y `metrics_calculator` tienen `save_output: true`.

Para activar modo debug completo (todos los steps guardan output), modificar `_create_cutie_pipeline_config()` en `ai-core/src/api/tasks.py` y poner `"save_output": True` en cada step.

---

## G. Benchmark Feature — Design Hooks

### Hooks existentes reutilizables

| Hook | Archivo/función | Reutilizable para bench |
|------|-----------------|------------------------|
| `process_video_task()` | `tasks.py` | Core de ejecución — puede llamarse directamente con cualquier `video_id` |
| `create_api_pipeline_config()` | `tasks.py` | Genera config YAML para cualquier combinación backend/modelos |
| `build_api_results()` | `tasks.py` | Construye el resultado desde `track_refiner` output + métricas |
| `StorageManager` | `storage.py` | Almacena/recupera jobs, results y artefactos |
| `VideoJob` dataclass | `storage.py` | Modelo de job — extensible con campos adicionales |
| `PipelineRunner` | `pipeline/runner.py` | Orquestador — acepta YAML arbitrary + callback de progreso |
| `_server_config` | `tasks.py` | Config runtime de modelos — ya soporta múltiples backends |
| `enqueue_processing()` | `tasks.py` | Encola job para procesamiento async |

### Carencias actuales para benchmark + propuestas

#### 1. Video source local (sin upload)

**Carencia:** Hoy el pipeline asume que el video fue subido a `data/api/uploads/{id}/input.mp4`. Para benchmark, queremos apuntar a un asset local del servidor (ej. `data/raw/video_test_1.mp4`) sin subir el archivo.

**Propuesta mínima:**
- Nuevo endpoint `POST /api/v1/bench/run` que acepte `body: {asset_path: "video_test_1.mp4", disc_center_x, ...}`
- `process_video_task()` ya crea symlink en `data/raw/` — el bench simplemente pasa el path directamente
- Cambios: crear `ai-core/src/api/routes/bench.py`, agregar `video_source: "local_asset"` como campo en `VideoJob`

#### 2. Endpoint de capabilities / handshake

**Carencia:** No existe un endpoint que describa qué backends están disponibles, qué modelos están cargados, y qué versión del contrato se usa.

**Propuesta mínima:**
- `GET /api/v1/bench/capabilities` o `GET /api/v1/info`
- Respuesta basada en `get_server_model_config()` + lista de backends disponibles
- Cambios: una función en `tasks.py` que devuelva capabilities, endpoint nuevo en `main.py`

#### 3. Ejecución batch

**Carencia:** Solo se puede enviar un video a la vez. Para benchmark, queremos enviar N videos (o N seeds para el mismo video) en un solo request y obtener una lista de `job_id`.

**Propuesta mínima:**
- `POST /api/v1/bench/run` acepta lista de jobs: `[{asset_path, seed, ...}, ...]`
- Itera sobre la lista, crea un `VideoJob` por item, los encola todos
- Responde con `{"jobs": ["id1", "id2", ...]}` y permite polling individual vía `/status`
- Cambios: modificar `create_job()` en `storage.py` para aceptar `source_type: "local_asset"`, `bench_run_id: str` y otros tags

#### 4. Etiquetado de runs de benchmark

**Carencia:** No hay forma de agrupar jobs bajo un "run de benchmark" (mismo dataset, misma configuración).

**Propuesta mínima:**
- Añadir campos opcionales a `VideoJob`: `bench_run_id: Optional[str]`, `tags: Optional[dict]`, `asset_name: Optional[str]`
- Cambios: `storage.py` → `VideoJob` dataclass + `create_job()` signature + `VideoJob.from_dict()`

#### 5. Comparación con ground-truth

**Carencia:** No existe estructura de almacenamiento ni endpoints para ground-truth.

**Propuesta mínima (diferida):**
- Nuevo directorio `data/benchmark/ground_truth/{asset_name}/gt.json`
- Schema de GT alineado con ResultsContract (mismas keys, valores manuales)
- `GET /api/v1/bench/{job_id}/compare` que carga GT y results y devuelve métricas de error
- Cambios: nuevo `bench_storage.py`, no toca `StorageManager` actual

### Para cada propuesta: dónde tocar

| Propuesta | Archivos a modificar |
|-----------|----------------------|
| Video source local | `storage.py` (VideoJob), `tasks.py` (process_video_task), `routes/bench.py` (nuevo) |
| Capabilities | `tasks.py` (nueva función), `main.py` (endpoint GET) |
| Batch | `routes/bench.py`, `storage.py`, `tasks.py` (enqueue_batch) |
| Etiquetado | `storage.py` (VideoJob + from_dict + create_job), `routes/bench.py` |
| Ground-truth | `bench_storage.py` (nuevo), `routes/bench.py` |

---

## H. Quickstart Reproducible

### Arrancar el servidor

```bash
# Prerrequisitos: uv instalado, Cutie weights en su lugar
# data/api/results/ y data/api/uploads/ se crean automáticamente

cd ai-core
PYTHONPATH=vendors/cutie:src:. uv run python control_panel.py
# Abre http://localhost:5001 → Control Panel
# Desde ahí: Start FastAPI Server → configura modelo → sube video

# O directo (sin control panel):
PYTHONPATH=vendors/cutie:src:. uv run python run_api.py
# Swagger: http://localhost:8000/docs
```

### End-to-end con curl

```bash
BASE=http://localhost:8000

# 1. Upload video con seed
RESPONSE=$(curl -s -X POST $BASE/api/v1/videos/upload \
  -F "file=@../data/raw/video_test_1.mp4" \
  -F "disc_center_x=469.5" \
  -F "disc_center_y=1438.3" \
  -F "disc_radius=127.8")
echo "$RESPONSE"
VIDEO_ID=$(echo $RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['video_id'])")
echo "Job: $VIDEO_ID"

# 2. Poll status
while true; do
  STATUS=$(curl -s $BASE/api/v1/videos/$VIDEO_ID/status)
  PROG=$(echo $STATUS | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'], d['progress'], d.get('message',''))")
  echo "$PROG"
  STATE=$(echo $STATUS | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  [[ "$STATE" == "completed" || "$STATE" == "failed" ]] && break
  sleep 3
done

# 3. Get results
curl -s $BASE/api/v1/videos/$VIDEO_ID/results | python3 -m json.tool | head -60

# 4. (Opcional) Borrar
# curl -X DELETE $BASE/api/v1/videos/$VIDEO_ID
```

### Inspeccionar artefactos en disco

```bash
VIDEO_ID=17fdbc3a-c5d  # reemplazar

# Config exacta usada
cat data/api/results/$VIDEO_ID/pipeline_config.yaml

# Métricas por frame (CSV)
head -5 data/api/results/$VIDEO_ID/metrics_calculator_output.csv

# Summary JSON
cat data/api/results/$VIDEO_ID/metrics_calculator_output_summary.json | python3 -m json.tool

# Track del disco (raw Cutie output, antes de smooth)
cat data/api/results/$VIDEO_ID/cutie_disc_tracking_output.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Frames: {len(d)}')
print('Frame 0 keys:', list(list(d.values())[0][0].keys()) if d else 'empty')
"

# Resultado final
cat data/api/results/$VIDEO_ID/results.json | python3 -c "
import sys, json
r = json.load(sys.stdin)
print('video_id:', r['video_id'])
print('tracks:', len(r['tracks']), 'classes:', [t['class_name'] for t in r['tracks']])
print('metrics frames:', len(r['metrics']['frames']))
print('summary:', r['summary'])
"
```

### Verificar config de modelos activa

```bash
# Ver config actual
curl -s http://localhost:8000/api/v1/config/models | python3 -m json.tool

# Cambiar a YOLO backend
curl -s -X POST http://localhost:8000/api/v1/config/models \
  -H "Content-Type: application/json" \
  -d '{"tracking_backend": "yolo", "enable_person_detection": false, "enable_pose_estimation": false}'
```

---

## I. Checklist de TODOs

### 🔴 Alta prioridad (bloquean benchmark)

- [ ] **[UNKNOWN/TODO]** `pipeline.log` en `data/api/results/{id}/` queda vacío. El logger del `PipelineRunner` escribe a `data/outputs/` (path distinto). Revisar `ai-core/src/pipeline/runner.py:_setup_logging()` y `process_video_task()` en `tasks.py` — pasar `output_dir` correcto al runner o redirigir logs.
- [ ] **[FALTA]** No existe endpoint `GET /api/v1/info` o `GET /api/v1/bench/capabilities` — necesario para que un cliente externo conozca backends disponibles y versión del contrato antes de lanzar un benchmark.
- [ ] **[FALTA]** `VideoJob` no tiene campos `bench_run_id`, `tags`, `asset_name` — necesarios para agrupar corridas de benchmark y correlacionar con ground-truth.

### 🟡 Media prioridad (mejoran ergonomía de benchmark)

- [ ] **[FALTA]** No hay soporte de `video_source: "local_asset"` — hoy obligatoriamente se hace upload del archivo aunque ya esté en disco.
- [ ] **[FALTA]** La config `_server_config` se resetea al reiniciar FastAPI. Para benchmark automatizado (sin control panel), sería útil cargarla desde un archivo de config o variables de entorno.
- [ ] **[MEJORA]** `GET /api/v1/videos` no devuelve `original_filename`, `tracking_backend`, ni `selection_data` — útil para listar runs de benchmark y ver sus metadatos.
- [ ] **[MEJORA]** El campo `lift_duration_s` en `summary` actualmente es `metadata.duration_s` (duración total del video), no la duración del movimiento detectado. Para benchmark esto es una métrica importante; se debería calcular como el rango de frames donde el disco se está moviendo.

### 🟢 Baja prioridad (mejoras de calidad)

- [ ] **[DOC]** Actualizar `API_VERSION` en `ai-core/src/api/main.py` (actualmente `"1.0.0"`) para reflejar el contrato v2.0 que implementa.
- [ ] **[DOC]** Documentar schema de `cutie_disc_tracking_output.json` — actualmente no hay doc del formato de `Dict[frame_idx, List[Detection]]`.
- [ ] **[DOC]** Detallar exactamente cómo usa `ModelTracker` el campo `initial_selection` (verificar `ai-core/src/analysis/model_tracker.py`).
- [ ] **[TEST]** No existen tests automáticos del endpoint `/results` — agregar un test que valide el schema contra `AnalysisResults` Pydantic.
- [ ] **[MEJORA]** `cleanup_old_jobs()` de `StorageManager` no se llama automáticamente — no hay tarea periódica ni cron. Para benchmark con muchos videos, agregar una llamada en startup o un endpoint admin.

---

## Diff Plan — Archivos a tocar para implementar benchmark

> Esta sección lista los archivos del repo que se modificarían o crearían para implementar `/api/v1/bench/*`. No incluye código, solo mapa de cambios.

```
# NUEVOS archivos
ai-core/src/api/routes/bench.py
    └─ POST /api/v1/bench/run        (acepta asset_path + seed + bench_run_id + tags)
    └─ GET  /api/v1/bench/runs       (lista runs agrupados por bench_run_id)
    └─ GET  /api/v1/bench/{job_id}/compare  (compara results contra GT si existe)

ai-core/src/api/bench_storage.py
    └─ Carga/guarda ground-truth desde data/benchmark/ground_truth/
    └─ Genera métricas de comparación (MAE de trayectoria, error en summary)

data/benchmark/
├── ground_truth/
│   └── {asset_name}/
│       └── gt.json   (mismo schema que ResultsContract, valores manuales)
└── runs/
    └── {bench_run_id}.json  (metadatos de run: timestamp, config, job_ids)

# MODIFICADOS
ai-core/src/api/storage.py
    └─ VideoJob: + bench_run_id, tags, asset_name, source_type
    └─ VideoJob.from_dict(): backward compat para nuevos campos
    └─ create_job(): nuevos params opcionales
    └─ list_jobs(): filtro opcional por bench_run_id

ai-core/src/api/tasks.py
    └─ process_video_task(): soporte source_type="local_asset" (no symlink, path directo)
    └─ Nueva función: get_capabilities() → dict con backends, contrato version, etc.

ai-core/src/api/main.py
    └─ include_router(bench_router, prefix="/api/v1")
    └─ GET /api/v1/info o /api/v1/bench/capabilities → get_capabilities()

ai-core/src/api/routes/__init__.py
    └─ Exportar bench_router

ai-core/src/api/models.py
    └─ BenchRunRequest (Pydantic model para POST /bench/run)
    └─ BenchCapabilities (Pydantic model para GET /capabilities)
    └─ BenchCompareResponse (diferencias vs GT)

docs/BENCHMARK_API.md   ← ESTE ARCHIVO
    └─ Actualizar secciones B, C, G con endpoints implementados
```
