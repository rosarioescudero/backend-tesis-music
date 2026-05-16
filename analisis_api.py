import json
import os
import re
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "tracking_audio_v6.py"
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

    pattern_pa = r"(S\d+)_([A-Z]+)_(PA)([+-]\d+)__(\d+)-(\d+)_([A-Za-z]+\d+)"
    match_pa = re.search(pattern_pa, normalized)
    if match_pa:
        porcentaje_cambio = int(match_pa.group(4))
        bpm_2 = 120 * (1 + porcentaje_cambio / 100)
        return {
            "sujeto": re.sub(r"\D", "", match_pa.group(1)),
            "mano": match_pa.group(2),
            "tipo_metronomo": "PA",
            "porcentaje_cambio": porcentaje_cambio,
            "bpm_1": 120,
            "bpm_2": round(bpm_2),
            "clics_1": int(match_pa.group(5)),
            "clics_2": int(match_pa.group(6)),
            "medicion": re.sub(r"\D", "", match_pa.group(7)),
        }

    pattern_pr = r"(S\d+)_([A-Z]+)_(PR)([+-]\d+)__(\d+)-(\d+)-(\d+)-([A-Za-z]+\d+)"
    match_pr = re.search(pattern_pr, normalized)
    if match_pr:
        porcentaje_cambio = int(match_pr.group(4))
        bpm_2 = 120 * (1 + porcentaje_cambio / 100)
        return {
            "sujeto": re.sub(r"\D", "", match_pr.group(1)),
            "mano": match_pr.group(2),
            "tipo_metronomo": "PR",
            "porcentaje_cambio": porcentaje_cambio,
            "bpm_1": 120,
            "bpm_2": round(bpm_2),
            "clics_1": int(match_pr.group(5)),
            "clics_rampa": int(match_pr.group(6)),
            "clics_2": int(match_pr.group(7)),
            "medicion": re.sub(r"\D", "", match_pr.group(8)),
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

    if cv2 is None:
        result["video"] = {
            "fps": None,
            "frames": None,
            "duracion_segundos": None,
            "resolucion": None,
        }
        return result

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
    else:
        result["video"] = {
            "fps": None,
            "frames": None,
            "duracion_segundos": None,
            "resolucion": None,
        }
    capture.release()
    return result


def get_missing_packages():
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append({"module": module_name, "package": package_name})
    return missing


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_real_analysis(video_path: Path, metronome_path: Path, output_dir: Path):
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"No existe el script de análisis: {SCRIPT_PATH}")

    clean_output_dir(output_dir)

    env = os.environ.copy()
    env.update({
        "ANALYSIS_VIDEO_PATH": str(video_path.resolve()),
        "ANALYSIS_METRONOME_PATH": str(metronome_path.resolve()),
        "ANALYSIS_OUTPUT_DIR": str(output_dir.resolve()),
        "ANALYSIS_SHOW_PLOTS": "0"
    })

    # Ejecutamos el script
    process = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
    )

    # 🔥 SI EL SCRIPT FALLA, ESTO NOS DIRÁ POR QUÉ EN RENDER
    if process.returncode != 0:
        print("--- ERROR DETALLADO DEL SCRIPT ---")
        print(process.stderr)
        print("---------------------------------")
        raise RuntimeError(f"El motor de análisis falló. Error: {process.stderr[:200]}")

    # 🔥 BUSCADOR DETECTIVE: Buscamos cualquier JSON generado
    json_files = list(output_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"El script terminó pero no generó resultados en {output_dir}")

    # Leemos el primer JSON que encontremos
    manifest_path = json_files[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest, {"stdout": process.stdout, "stderr": process.stderr}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "analisis_api"})


@app.route("/analizar", methods=["POST"])
@app.route("/analizar", methods=["POST"])
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
            
        # Si no viene el nombre del archivo, lo extraemos de la URL
        if not nombre_archivo:
            nombre_archivo = ruta_video_original.split("/")[-1]
            
        # 1. 🔥 CAMBIO CLAVE: Descargamos el video con su NOMBRE REAL ORIGINAL
        video_dir = BASE_DIR / "temp_videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        ruta_video_local = video_dir / nombre_archivo
        
        print(f"DEBUG: Descargando video con su nombre real en {ruta_video_local}...")
        response = requests.get(ruta_video_original, stream=True)
        response.raise_for_status()
        
        with open(ruta_video_local, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("DEBUG: Video descargado con éxito.")

        # 2. BUSCADOR DE AUDIO INTELIGNETE (Soporta ISO, PA y PR)
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
            print(f"ERROR: No se encontró el metrónomo {nombre_audio} en ninguna carpeta.")
            return jsonify({"status": "error", "message": f"Audio no encontrado: {nombre_audio}"}), 404

        # 3. Carpeta de salida local en Python
        output_dir = BASE_DIR / "analysis_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 4. Ejecutamos el análisis pasando el archivo con el nombre correcto
        print(f"DEBUG: Ejecutando motor con metrónomo: {metronome_path}")
        manifest, logs = run_real_analysis(ruta_video_local, metronome_path, output_dir)

        # Limpieza de seguridad para no llenar el almacenamiento de Render
        if ruta_video_local.exists():
            ruta_video_local.unlink()

        return jsonify({
            "status": "success",
            "mensaje": "Análisis completado",
            "metadata_extraida": manifest.get("metadata") or extract_metadata_from_filename(nombre_archivo),
            "analisis": manifest,
            "logs_python": logs["stderr"][-500:]
        })

    except Exception as error:
        print("--- CRASH EN EL BACKEND ---")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Error interno: {str(error)}"}), 500


if __name__ == "__main__":
    import os
    print("Servidor de Python activo")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
