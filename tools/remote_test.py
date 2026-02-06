#!/usr/bin/env python3
"""
remote_test.py - Test remoto del pipeline de análisis de video
==============================================================

Script standalone para enviar un video al servidor de análisis
y recibir los resultados. Diseñado para ejecutarse desde cualquier
PC sin necesidad de clonar el repositorio completo.

REQUISITOS:
    pip install requests

USO:
    python remote_test.py --url https://TU-URL.ngrok-free.app --video video.mp4

    # Con selección de disco (recomendado)
    python remote_test.py --url https://TU-URL.ngrok-free.app --video video.mp4 \
        --disc-x 470 --disc-y 1436 --disc-radius 123

    # Guardar resultados en archivo específico
    python remote_test.py --url https://... --video video.mp4 --output results.json

EJEMPLOS:
    # Test básico
    python remote_test.py -u https://abc123.ngrok-free.app -v mi_video.mp4

    # Con todos los parámetros
    python remote_test.py \
        --url https://abc123.ngrok-free.app \
        --video levantamiento.mp4 \
        --disc-x 587 --disc-y 623 --disc-radius 74 \
        --output analysis_results.json \
        --timeout 300

NOTAS:
    - El servidor debe estar corriendo (FastAPI + Ngrok si es remoto)
    - Los valores de disc-x, disc-y, disc-radius son coordenadas en píxeles
      del centro y radio del disco en el primer frame del video
    - Sin parámetros de disco, el sistema usa auto-detección (menos preciso)
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ============================================================
# Única dependencia externa
# ============================================================
try:
    import requests
except ImportError:
    print("ERROR: Necesitas instalar 'requests'")
    print("       pip install requests")
    sys.exit(1)


# ============================================================
# Configuración
# ============================================================
VERSION = "1.0.0"
DEFAULT_TIMEOUT = 120  # segundos para upload
POLL_INTERVAL = 2      # segundos entre checks de status


def print_banner():
    """Muestra banner inicial."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           🏋️ Remote Video Analysis Test v{version}              ║
╚═══════════════════════════════════════════════════════════════╝
""".format(version=VERSION))


def check_server(base_url: str, headers: dict) -> bool:
    """Verifica que el servidor está respondiendo."""
    try:
        r = requests.get(f"{base_url}/health", headers=headers, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"   Error de conexión: {e}")
        return False


def upload_video(
    base_url: str,
    video_path: str,
    disc_x: float = None,
    disc_y: float = None,
    disc_radius: float = None,
    headers: dict = None,
    timeout: int = DEFAULT_TIMEOUT
) -> dict:
    """
    Sube un video al servidor para procesamiento.
    
    Returns:
        dict con video_id y status, o None si falla
    """
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"   ERROR: Video no encontrado: {video_path}")
        return None
    
    file_size_mb = video_file.stat().st_size / (1024 * 1024)
    print(f"   Archivo: {video_file.name} ({file_size_mb:.1f} MB)")
    
    # Preparar form data
    with open(video_path, 'rb') as f:
        files = {'file': (video_file.name, f, 'video/mp4')}
        data = {}
        
        if disc_x is not None and disc_y is not None and disc_radius is not None:
            data['disc_center_x'] = disc_x
            data['disc_center_y'] = disc_y
            data['disc_radius'] = disc_radius
            print(f"   Selección disco: center=({disc_x}, {disc_y}), radius={disc_radius}")
        else:
            print("   Sin selección de disco (auto-detección)")
        
        try:
            r = requests.post(
                f"{base_url}/api/v1/videos/upload",
                headers=headers,
                files=files,
                data=data,
                timeout=timeout
            )
        except requests.Timeout:
            print(f"   ERROR: Timeout después de {timeout}s")
            return None
        except Exception as e:
            print(f"   ERROR: {e}")
            return None
    
    if r.status_code != 200:
        print(f"   ERROR: Server respondió {r.status_code}")
        print(f"   {r.text[:200]}")
        return None
    
    return r.json()


def wait_for_processing(
    base_url: str,
    video_id: str,
    headers: dict,
    max_wait: int = 600  # 10 minutos máximo
) -> str:
    """
    Espera a que el procesamiento termine.
    
    Returns:
        'completed', 'failed', o 'timeout'
    """
    start_time = time.time()
    last_step = ""
    
    while (time.time() - start_time) < max_wait:
        try:
            r = requests.get(
                f"{base_url}/api/v1/videos/{video_id}/status",
                headers=headers,
                timeout=10
            )
        except Exception as e:
            print(f"   ⚠️ Error de conexión: {e}")
            time.sleep(POLL_INTERVAL)
            continue
        
        if r.status_code != 200:
            print(f"   ⚠️ Status check: {r.status_code}")
            time.sleep(POLL_INTERVAL)
            continue
        
        status = r.json()
        state = status.get('status', 'unknown')
        progress = status.get('progress', 0) * 100
        step = status.get('current_step', '')
        
        # Mostrar progreso solo si cambió
        if step != last_step or progress % 25 < 5:
            elapsed = time.time() - start_time
            print(f"   [{progress:5.1f}%] {state}: {step} ({elapsed:.0f}s)")
            last_step = step
        
        if state == 'completed':
            return 'completed'
        elif state == 'failed':
            print(f"   ❌ Error: {status.get('message', 'Unknown error')}")
            return 'failed'
        
        time.sleep(POLL_INTERVAL)
    
    print(f"   ⏰ Timeout después de {max_wait}s")
    return 'timeout'


def get_results(base_url: str, video_id: str, headers: dict) -> dict:
    """Obtiene los resultados del procesamiento."""
    try:
        r = requests.get(
            f"{base_url}/api/v1/videos/{video_id}/results",
            headers=headers,
            timeout=30
        )
        if r.status_code == 200:
            return r.json()
        else:
            print(f"   ERROR: {r.status_code} - {r.text[:100]}")
            return None
    except Exception as e:
        print(f"   ERROR: {e}")
        return None


def print_results_summary(results: dict):
    """Muestra un resumen de los resultados."""
    print("\n" + "="*60)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*60)
    
    # Metadata
    metadata = results.get('metadata', {})
    print(f"\n📹 Video:")
    print(f"   • FPS: {metadata.get('fps', 'N/A')}")
    print(f"   • Resolución: {metadata.get('width', '?')}x{metadata.get('height', '?')}")
    print(f"   • Duración: {metadata.get('duration_s', 0):.2f}s")
    print(f"   • Frames: {metadata.get('total_frames', 'N/A')}")
    
    # Tracks
    tracks = results.get('tracks', [])
    print(f"\n🎯 Objetos trackeados: {len(tracks)}")
    for t in tracks:
        frames_count = len(t.get('frames', {}))
        trajectory_len = len(t.get('trajectory', []))
        print(f"   • Track {t.get('track_id', '?')}: {t.get('class_name', '?')}")
        print(f"     - Frames con detección: {frames_count}")
        print(f"     - Puntos en trayectoria: {trajectory_len}")
    
    # Métricas
    metrics = results.get('metrics', {})
    summary = results.get('summary', {})
    
    if summary:
        print(f"\n📈 Métricas del movimiento:")
        print(f"   • Velocidad pico: {summary.get('peak_speed_m_s', 0):.2f} m/s")
        print(f"   • Potencia pico:  {summary.get('peak_power_w', 0):.0f} W")
        print(f"   • Altura máxima:  {summary.get('max_height_m', 0):.2f} m")
        print(f"   • Altura mínima:  {summary.get('min_height_m', 0):.2f} m")
    
    if metrics:
        frames = metrics.get('frames', [])
        if frames:
            print(f"\n📐 Serie temporal:")
            print(f"   • Frames analizados: {len(frames)} ({frames[0]} - {frames[-1]})")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Test remoto del pipeline de análisis de video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s -u https://abc.ngrok-free.app -v video.mp4
  %(prog)s -u https://abc.ngrok-free.app -v video.mp4 --disc-x 470 --disc-y 1436 --disc-radius 123
        """
    )
    
    parser.add_argument(
        '-u', '--url',
        required=True,
        help='URL del servidor (ej: https://abc123.ngrok-free.app)'
    )
    parser.add_argument(
        '-v', '--video',
        required=True,
        help='Ruta al archivo de video'
    )
    parser.add_argument(
        '--disc-x',
        type=float,
        help='Coordenada X del centro del disco (píxeles)'
    )
    parser.add_argument(
        '--disc-y',
        type=float,
        help='Coordenada Y del centro del disco (píxeles)'
    )
    parser.add_argument(
        '--disc-radius',
        type=float,
        help='Radio del disco (píxeles)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Archivo donde guardar resultados JSON'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f'Timeout para upload en segundos (default: {DEFAULT_TIMEOUT})'
    )
    parser.add_argument(
        '--max-wait',
        type=int,
        default=600,
        help='Tiempo máximo de espera para procesamiento (default: 600s)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {VERSION}'
    )
    
    args = parser.parse_args()
    
    # Headers para Ngrok
    headers = {
        "ngrok-skip-browser-warning": "true"
    }
    
    # Banner
    print_banner()
    
    # Info
    print(f"🎯 Servidor: {args.url}")
    print(f"📹 Video: {args.video}")
    print()
    
    # ============================================================
    # Paso 1: Verificar servidor
    # ============================================================
    print("1️⃣  Verificando servidor...")
    if not check_server(args.url, headers):
        print("   ❌ No se puede conectar al servidor")
        print("\n   Verifica que:")
        print("   • El servidor FastAPI está corriendo")
        print("   • La URL de Ngrok es correcta")
        print("   • Tienes conexión a internet")
        sys.exit(1)
    print("   ✅ Servidor disponible")
    
    # ============================================================
    # Paso 2: Subir video
    # ============================================================
    print("\n2️⃣  Subiendo video...")
    upload_result = upload_video(
        args.url,
        args.video,
        disc_x=args.disc_x,
        disc_y=args.disc_y,
        disc_radius=args.disc_radius,
        headers=headers,
        timeout=args.timeout
    )
    
    if not upload_result:
        print("   ❌ Error en upload")
        sys.exit(1)
    
    video_id = upload_result.get('video_id')
    print(f"   ✅ Video subido! ID: {video_id}")
    
    # ============================================================
    # Paso 3: Esperar procesamiento
    # ============================================================
    print("\n3️⃣  Esperando procesamiento...")
    status = wait_for_processing(args.url, video_id, headers, args.max_wait)
    
    if status != 'completed':
        print(f"\n   ❌ Procesamiento terminó con estado: {status}")
        sys.exit(1)
    print("   ✅ Procesamiento completado!")
    
    # ============================================================
    # Paso 4: Obtener resultados
    # ============================================================
    print("\n4️⃣  Obteniendo resultados...")
    results = get_results(args.url, video_id, headers)
    
    if not results:
        print("   ❌ Error obteniendo resultados")
        sys.exit(1)
    
    # Guardar resultados
    output_file = args.output or f"results_{video_id}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✅ Resultados guardados: {output_file}")
    
    # ============================================================
    # Paso 5: Mostrar resumen
    # ============================================================
    print_results_summary(results)
    
    print("\n✅ Test completado exitosamente!")
    print(f"\n💡 El archivo {output_file} contiene todos los datos:")
    print("   • tracks: trayectorias de objetos detectados")
    print("   • metrics: series temporales (velocidad, altura, potencia)")
    print("   • summary: valores pico y estadísticas")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
