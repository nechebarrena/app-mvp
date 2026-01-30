# FastAPI - Guía Rápida

## Iniciar el Servidor

```bash
cd ai-core
PYTHONPATH=src:. uv run python run_api.py
```

El servidor estará disponible en:
- **API**: http://localhost:8000
- **Swagger UI** (probar endpoints): http://localhost:8000/docs
- **ReDoc** (documentación): http://localhost:8000/redoc

### Opciones

```bash
# Con auto-reload (desarrollo)
PYTHONPATH=src:. uv run python run_api.py --reload

# Puerto diferente
PYTHONPATH=src:. uv run python run_api.py --port 9000

# Acceso desde otros dispositivos en la red
PYTHONPATH=src:. uv run python run_api.py --host 0.0.0.0
```

---

## Uso Básico (curl)

### 1. Subir un video

```bash
# Upload simple (sin selección de disco)
curl -X POST "http://localhost:8000/api/v1/videos/upload" \
  -F "file=@/path/to/video.mp4"

# Upload CON selección de disco (recomendado para mejor tracking)
curl -X POST "http://localhost:8000/api/v1/videos/upload" \
  -F "file=@/path/to/video.mp4" \
  -F "disc_center_x=587" \
  -F "disc_center_y=623" \
  -F "disc_radius=74"
```

**Parámetros de selección del disco** (opcionales pero recomendados):
- `disc_center_x`: Coordenada X del centro del disco en el frame inicial (píxeles)
- `disc_center_y`: Coordenada Y del centro del disco en el frame inicial (píxeles)
- `disc_radius`: Radio del disco en el frame inicial (píxeles)

Estos valores activan heurísticas de tracking single-object que mejoran significativamente la calidad del tracking.

Respuesta:
```json
{"video_id": "abc123", "status": "pending", "message": "..."}
```

### 2. Verificar estado

```bash
curl "http://localhost:8000/api/v1/videos/{video_id}/status"
```

Estados posibles: `pending` → `processing` → `completed` (o `failed`)

### 3. Obtener resultados

```bash
curl "http://localhost:8000/api/v1/videos/{video_id}/results"
```

### 4. Eliminar video

```bash
curl -X DELETE "http://localhost:8000/api/v1/videos/{video_id}"
```

---

## Ejemplo Completo

```bash
# 1. Subir video
VIDEO_ID=$(curl -s -X POST "http://localhost:8000/api/v1/videos/upload" \
  -F "file=@../data/raw/video_test_1.mp4" | python3 -c "import sys, json; print(json.load(sys.stdin)['video_id'])")

echo "Video ID: $VIDEO_ID"

# 2. Esperar procesamiento (~30-60 segundos)
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/v1/videos/$VIDEO_ID/status" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then break; fi
  sleep 3
done

# 3. Ver resultados
curl -s "http://localhost:8000/api/v1/videos/$VIDEO_ID/results" | python3 -m json.tool
```

---

## Estructura de Respuesta `/results`

```json
{
  "video_id": "abc123",
  "status": "completed",
  "metadata": {
    "fps": 29.5,
    "width": 1080,
    "height": 1920,
    "duration_s": 3.9,
    "total_frames": 115
  },
  "tracks": [
    {
      "track_id": 1,
      "class_name": "person",
      "frames": {
        "0": {"bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 800}, "confidence": 0.95},
        "1": {...}
      },
      "trajectory": [[200, 500], [202, 498], ...]
    },
    {
      "track_id": 2,
      "class_name": "frisbee",
      "frames": {...},
      "trajectory": [...]
    }
  ],
  "metrics": {
    "frames": [0, 1, 2, ...],
    "time_s": [0.0, 0.033, 0.067, ...],
    "height_m": [...],
    "speed_m_s": [...],
    "power_w": [...]
  },
  "summary": {
    "peak_speed_m_s": 5.87,
    "peak_power_w": 14376,
    "max_height_m": 0.82,
    "min_height_m": -0.37
  }
}
```

---

## Acceso desde Móvil (misma red WiFi)

1. Obtener IP del servidor:
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   # Ejemplo: 192.168.1.100
   ```

2. Iniciar servidor con acceso externo:
   ```bash
   PYTHONPATH=src:. uv run python run_api.py --host 0.0.0.0
   ```

3. Desde el móvil, acceder a:
   - `http://192.168.1.100:8000/docs` (probar en navegador)
   - Usar esa URL como base en la app

---

## Troubleshooting

### Puerto en uso
```bash
lsof -ti:8000 | xargs kill -9
```

### Ver logs
```bash
PYTHONPATH=src:. uv run python run_api.py 2>&1 | tee api.log
```

### Video no detectado
- Asegurarse que el video tiene objetos detectables (persona, disco)
- El modelo YOLO COCO detecta: person, frisbee, sports ball
- Para discos de barbell, usar modelo custom

---

## Script de Test Completo

Para probar el flujo completo (selección de disco + upload + resultados):

**IMPORTANTE:** El servidor debe estar corriendo en una terminal separada.

### Paso 1: Iniciar el servidor (en una terminal)
```bash
cd ai-core
PYTHONPATH=src:. uv run python run_api.py
```

### Paso 2: Ejecutar el test (en otra terminal)
```bash
cd ai-core
PYTHONPATH=src:. uv run python test_api_full.py --video ../data/raw/video_test_1.mp4
```

Este script:
1. Verifica que el servidor esté corriendo
2. Abre la herramienta GUI de selección de disco
3. Sube el video con la selección al API
4. Espera el procesamiento
5. Muestra los resultados

### Opciones adicionales

```bash
# Usar una selección existente (sin abrir la GUI)
PYTHONPATH=src:. uv run python test_api_full.py --video ../data/raw/video_test_1.mp4 --skip-selection

# Especificar archivo de selección
PYTHONPATH=src:. uv run python test_api_full.py --video ../data/raw/video_test_1.mp4 --skip-selection --selection-file /tmp/my_selection.json

# Abrir el visualizador automáticamente al terminar
PYTHONPATH=src:. uv run python test_api_full.py --video ../data/raw/video_test_1.mp4 --launch-viewer
```

---

## Visualización de Resultados API

El visualizador interactivo soporta tanto salida del pipeline como JSON de la API:

```bash
# Visualizar resultados de la API
cd ai-core
PYTHONPATH=src:. uv run python view_analysis.py ../data/api/results/{video_id}/results.json

# El visualizador detecta automáticamente el tipo de fuente y carga:
# - Video original (desde uploads/)
# - Métricas desde el JSON
```

Esto permite validar visualmente los resultados de la API antes de integrar en la app móvil.

---

## Archivos Clave

| Archivo | Descripción |
|---------|-------------|
| `run_api.py` | Script para iniciar el servidor |
| `test_api_full.py` | Script de test con selección de disco |
| `src/api/main.py` | Aplicación FastAPI principal |
| `src/api/routes/videos.py` | Endpoints de video |
| `src/api/tasks.py` | Procesamiento en background |
| `src/api/storage.py` | Gestión de archivos |
