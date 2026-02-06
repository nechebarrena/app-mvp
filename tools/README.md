# Tools - Scripts Standalone

Herramientas independientes que pueden descargarse y ejecutarse sin necesidad de clonar el repositorio completo.

---

## remote_test.py

Script para probar el pipeline de análisis de video de forma remota.

### Requisitos

```bash
pip install requests
```

### Uso Básico

```bash
# Descargar el script
curl -O https://raw.githubusercontent.com/nechebarrena/app-mvp/main/tools/remote_test.py

# Ejecutar (sin selección de disco)
python remote_test.py \
  --url https://TU-URL.ngrok-free.app \
  --video mi_video.mp4
```

### Uso con Selección de Disco (Recomendado)

Para mejor precisión en el tracking, proporciona las coordenadas del disco en el primer frame:

```bash
python remote_test.py \
  --url https://TU-URL.ngrok-free.app \
  --video mi_video.mp4 \
  --disc-x 470 \
  --disc-y 1436 \
  --disc-radius 123
```

### Obtener Coordenadas del Disco

Las coordenadas se obtienen del primer frame del video:
- `--disc-x`: Posición X del centro del disco (píxeles desde la izquierda)
- `--disc-y`: Posición Y del centro del disco (píxeles desde arriba)
- `--disc-radius`: Radio del disco (píxeles)

**Métodos para obtenerlas:**

1. **Visualmente:** Abre el video, pausa en frame 0, estima las coordenadas
2. **Con herramienta GUI (en Mac con el repo):**
   ```bash
   cd ai-core
   PYTHONPATH=src:. uv run python select_disc.py video.mp4 /tmp/coords.json
   cat /tmp/coords.json
   # {"center": [470, 1436], "radius": 123, ...}
   ```

### Opciones Completas

```
usage: remote_test.py [-h] -u URL -v VIDEO [--disc-x DISC_X] [--disc-y DISC_Y]
                      [--disc-radius DISC_RADIUS] [-o OUTPUT] [--timeout TIMEOUT]
                      [--max-wait MAX_WAIT] [--version]

Opciones:
  -u, --url URL         URL del servidor (requerido)
  -v, --video VIDEO     Ruta al video (requerido)
  --disc-x DISC_X       Coordenada X del centro del disco
  --disc-y DISC_Y       Coordenada Y del centro del disco
  --disc-radius RADIUS  Radio del disco en píxeles
  -o, --output FILE     Archivo para guardar JSON de resultados
  --timeout SECONDS     Timeout para upload (default: 120)
  --max-wait SECONDS    Tiempo máximo de procesamiento (default: 600)
```

### Ejemplo de Salida

```
╔═══════════════════════════════════════════════════════════════╗
║           🏋️ Remote Video Analysis Test v1.0.0              ║
╚═══════════════════════════════════════════════════════════════╝

🎯 Servidor: https://abc123.ngrok-free.app
📹 Video: video_test.mp4

1️⃣  Verificando servidor...
   ✅ Servidor disponible

2️⃣  Subiendo video...
   Archivo: video_test.mp4 (15.3 MB)
   Selección disco: center=(470, 1436), radius=123
   ✅ Video subido! ID: f3a1b2c3

3️⃣  Esperando procesamiento...
   [  0.0%] processing: yolo_coco_detection (5s)
   [ 35.0%] processing: detection_filter (12s)
   [ 55.0%] processing: disc_tracking (18s)
   [ 75.0%] processing: track_refiner (22s)
   [100.0%] completed: extracting_results (28s)
   ✅ Procesamiento completado!

4️⃣  Obteniendo resultados...
   ✅ Resultados guardados: results_f3a1b2c3.json

============================================================
📊 RESUMEN DE RESULTADOS
============================================================

📹 Video:
   • FPS: 29.97
   • Resolución: 1080x1920
   • Duración: 3.87s
   • Frames: 116

🎯 Objetos trackeados: 2
   • Track 1: frisbee
     - Frames con detección: 114
     - Puntos en trayectoria: 114
   • Track 2: person
     - Frames con detección: 116
     - Puntos en trayectoria: 116

📈 Métricas del movimiento:
   • Velocidad pico: 2.45 m/s
   • Potencia pico:  1850 W
   • Altura máxima:  0.82 m
   • Altura mínima:  -0.15 m

============================================================

✅ Test completado exitosamente!
```

### Archivo de Resultados

El JSON generado contiene:

```json
{
  "video_id": "f3a1b2c3",
  "status": "completed",
  "metadata": {
    "fps": 29.97,
    "width": 1080,
    "height": 1920,
    "duration_s": 3.87,
    "total_frames": 116
  },
  "tracks": [...],
  "metrics": {
    "frames": [0, 1, 2, ...],
    "time_s": [0.0, 0.033, ...],
    "height_m": [...],
    "speed_m_s": [...],
    "power_w": [...]
  },
  "summary": {
    "peak_speed_m_s": 2.45,
    "peak_power_w": 1850,
    "max_height_m": 0.82,
    "min_height_m": -0.15
  }
}
```

---

## Solución de Problemas

### "No se puede conectar al servidor"

1. Verificar que FastAPI está corriendo en la Mac
2. Verificar que Ngrok está activo y la URL es correcta
3. La URL de Ngrok cambia cada vez que se reinicia

### "Timeout en upload"

- Videos grandes pueden tardar más
- Usa `--timeout 300` para dar más tiempo

### "Processing failed"

- Revisar logs en el Control Panel de la Mac
- El modelo puede no detectar objetos si el video es muy diferente al entrenamiento
