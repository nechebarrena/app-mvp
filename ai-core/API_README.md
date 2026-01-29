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
curl -X POST "http://localhost:8000/api/v1/videos/upload" \
  -F "file=@/path/to/video.mp4"
```

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

## Archivos Clave

| Archivo | Descripción |
|---------|-------------|
| `run_api.py` | Script para iniciar el servidor |
| `src/api/main.py` | Aplicación FastAPI principal |
| `src/api/routes/videos.py` | Endpoints de video |
| `src/api/tasks.py` | Procesamiento en background |
| `src/api/storage.py` | Gestión de archivos |
