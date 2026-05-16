import json
import os
import re
import shutil
import sys
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

REQUIRED_PACKAGES = {
    "cv2": "opencv-python==4.8.0.74",
    "mediapipe": "mediapipe==0.10.14",
    "numpy": "numpy==1.26.4",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "librosa": "librosa",
    "flask": "flask",
    "flask_cors": "flask-cors",
}

def normalize_filename(filename: str) -> str:
    base = os.path.basename(filename or "")
    base = base.replace("\ufffd", "_").replace(" ", "_")
    return re.sub(r"^pista_\d+_", "", base, flags=re.IGNORECASE)

def extract_metadata_from_filename(filename: str) -> dict:
    normalized = normalize_filename(filename)
    pattern_iso = r"(S\d+)_([A-Z]+)_ISO(\d+)_([A-Za-z]+\d+)"
    match_iso = re.search(pattern_iso, normalized)
    if match_iso:
        return {
            "sujeto": re.sub(r"\D", "", match_iso.group(1)),
            "mano": match_iso.group(2),
            "tipo_metronomo": "ISO",
            "bpm": int(match_iso.group(3)),
            "medicion": re.sub(r"\D", "", match_iso.group(4)),
        }
    return {}

def inspect_video(path: Path) -> dict:
    stat = path.stat()
    result = {
        "archivo": path.name,
        "ruta_video": str(path.resolve()),
        "tamano_bytes": stat.st_size,
        "tamano_mb": round(stat.st_size / (1024 * 1024), 2),
    }
    if cv2 is not None:
        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_seconds = round(frame_count / fps, 3) if fps > 0 else None
            result["video"] = {
                "fps": round(fps, 3) if fps > 0 else None,
                "frames": frame_count,
                "duracion_segundos": duration_seconds,
                "resolucion": {"ancho": width, "alto": height},
            }
        capture.release()
    return result

def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

def run_real_analysis(video_path: Path, metronome_path: Path, output_dir: Path):
    # 🔥 IMPORTACIÓN INTERNA MAESTRA: Ahorra 250MB de RAM compartiendo el proceso
    import tracking_audio_v6
    clean_output_dir(output_dir)
    config = {
        'GRAFICAR': False,
        'OUTPUT_DIR': str(output_dir.resolve())
    }
    print("DEBUG: Ejecutando motor de análisis internamente...")
    manifest = tracking_audio_v6.run_analysis(
        video_path=str(video_path.resolve()),
        metronome_path=str(metronome_path.resolve()),
        config=config
    )
    if not manifest:
        raise RuntimeError("El motor de análisis no retornó resultados.")
    return manifest, {"stdout": "OK", "stderr": ""}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "analisis_api"})

@app.route("/analizar", methods=["POST"])
def analizar():
    try:
        import traceback
        datos = request.get_json(silent=True) or {}
        ruta_video_original = datos.get("ruta_video")
        ruta_audio_metronomo = datos.get("ruta_audio_metronomo")
        nombre_archivo = datos.get("nombre_archivo")
        print(f"DEBUG: Iniciando análisis para {ruta_video_original}")

        import requests
        
        if not ruta_video_original or not ruta_video_original.startswith("http"):
            return jsonify({"status": "error", "message": "Falta URL del video"}), 400
            
        if not nombre_archivo:
            nombre_archivo = ruta_video_original.split("/")[-1]

        # Descargamos el video eficientemente con nombre real
        video_dir = BASE_DIR / "temp_videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        ruta_video_local = video_dir / nombre_archivo
        
        print(f"DEBUG: Descargando video en {ruta_video_local}...")
        response = requests.get(ruta_video_original, stream=True)
        response.raise_for_status()
        
        with open(ruta_video_local, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Buscador inteligente de metrónomos
        nombre_audio = Path(ruta_audio_metronomo).name
        carpeta_met = nombre_audio.split("__")[0] if "__" in nombre_audio else ""
        posibles_rutas = [
            BASE_DIR / "audios" / "rampa" / carpeta_met / nombre_audio,
            BASE_DIR / "audios" / "isocronos" / nombre_audio,
            BASE_DIR / "audios" / "abruptos" / carpeta_met / nombre_audio,
            BASE_DIR / "audios" / nombre_audio
        ]
        
        metronome_path = None
        for p in posibles_rutas:
            if p.exists():
                metronome_path = p
                break
        
        if not metronome_path:
            return jsonify({"status": "error", "message": f"Audio no encontrado: {nombre_audio}"}), 404

        output_dir = BASE_DIR / "analysis_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        inspection = inspect_video(ruta_video_local)
        manifest, logs = run_real_analysis(ruta_video_local, metronome_path, output_dir)

        # Escaneamos las imágenes generadas por el script
        png_files = list(output_dir.glob("*.png"))
        public_graficos = [f"https://backend-tesis-music.onrender.com/results/{f.name}" for f in png_files]

        if ruta_video_local.exists():
            ruta_video_local.unlink()

        if isinstance(manifest, dict) and "video" not in manifest:
            manifest["video"] = inspection.get("video", {})

        return jsonify({
            "status": "success",
            "mensaje": "Análisis completado",
            "archivo": nombre_archivo,
            "metadata_extraida": manifest.get("metadata") or extract_metadata_from_filename(nombre_archivo),
            "analisis": manifest,
            "graficos": public_graficos,
            "generated_files": []
        })

    except Exception as error:
        print("--- CRASH EN EL BACKEND ---")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Error interno: {str(error)}"}), 500

@app.route("/results/<filename>")
def get_result_file(filename):
    return send_from_directory(BASE_DIR / "analysis_results", filename)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
