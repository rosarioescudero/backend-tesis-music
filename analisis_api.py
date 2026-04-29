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
SCRIPT_PATH = BASE_DIR / "analyze_video.py"
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
    env.update(
        {
            "ANALYSIS_VIDEO_PATH": str(video_path.resolve()),
            "ANALYSIS_METRONOME_PATH": str(metronome_path.resolve()),
            "ANALYSIS_OUTPUT_DIR": str(output_dir.resolve()),
            "ANALYSIS_INPUT_DIR": str(video_path.parent.resolve()),
            "ANALYSIS_SHOW_PLOTS": "0",
            "ANALYSIS_ENABLE_SOURCE_SEPARATION": "0",
        }
    )

    process = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
    )

    manifest_path = output_dir / "analysis_results.json"
    if process.returncode != 0:
        stdout_tail = process.stdout[-2500:] if process.stdout else ""
        stderr_tail = process.stderr[-2500:] if process.stderr else ""
        raise RuntimeError(
            "El script de análisis falló.\n"
            f"STDOUT (final):\n{stdout_tail}\n\nSTDERR (final):\n{stderr_tail}"
        )

    if not manifest_path.exists():
        stdout_tail = process.stdout[-2500:] if process.stdout else ""
        stderr_tail = process.stderr[-2500:] if process.stderr else ""
        raise RuntimeError(
            "El script terminó pero no generó analysis_results.json.\n"
            f"STDOUT (final):\n{stdout_tail}\n\nSTDERR (final):\n{stderr_tail}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logs = {
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    return manifest, logs


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "analisis_api"})


@app.route("/analizar", methods=["POST"])
def analizar():
    try:
        datos = request.get_json(silent=True) or {}
        ruta_video = datos.get("ruta_video")
        print("DEBUG ruta_video ANTES:", ruta_video)
        ruta_audio_metronomo = datos.get("ruta_audio_metronomo")

        import requests
        import tempfile
        
        # Si es URL, descargar el video
        if isinstance(ruta_video, str) and ruta_video.startswith("http"):
            response = requests.get(ruta_video)
            
            if response.status_code != 200:
                return jsonify({
                    "status": "error",
                    "message": f"No se pudo descargar el video desde la URL: {ruta_video}"
                }), 400
        
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            
            with open(temp_file.name, "wb") as f:
                f.write(response.content)
            
            ruta_video = temp_file.name
            print("DEBUG ruta_video DESPUÉS:", ruta_video)
        nombre_archivo = datos.get("nombre_archivo")
        output_dir_raw = datos.get("output_dir")

        if not ruta_video:
            return jsonify({"status": "error", "message": "Falta 'ruta_video' en la petición."}), 400
        if not ruta_audio_metronomo:
            return jsonify({"status": "error", "message": "Falta 'ruta_audio_metronomo' en la petición."}), 400

        video_path = Path(ruta_video)

        # 🔥 LIMPIAR NOMBRE DEL AUDIO (CLAVE)
        from pathlib import Path as PathLib
        nombre_audio = PathLib(ruta_audio_metronomo).name
        
        # 🔥 EXTRAER CARPETA (ej: PR+10)
        carpeta = nombre_audio.split("__")[0]
        
        # 🔥 RUTA CORRECTA CON SUBCARPETA
        metronome_path = BASE_DIR / "audios" / "rampa" / carpeta / nombre_audio
        
        output_dir = Path(output_dir_raw) if output_dir_raw else (BASE_DIR / "analysis_outputs" / video_path.stem)
        
        # DEBUG (opcional pero recomendable)
        print("DEBUG audio final:", metronome_path)

        if not video_path.exists():
            return jsonify({"status": "error", "message": f"No existe el video: {video_path}"}), 404
        if not metronome_path.exists():
            return jsonify({"status": "error", "message": f"No existe el audio del metrónomo: {metronome_path}"}), 404
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            return jsonify({"status": "error", "message": "El archivo recibido no tiene una extensión de video soportada."}), 400

        missing_packages = get_missing_packages()
        if missing_packages:
            paquetes = " ".join(item["package"] for item in missing_packages)
            return jsonify(
                {
                    "status": "error",
                    "message": (
                        "Faltan dependencias de Python para ejecutar el análisis real. "
                        f"Módulos ausentes: {', '.join(item['module'] for item in missing_packages)}."
                    ),
                    "missing_packages": missing_packages,
                    "install_command": f'"{sys.executable}" -m pip install -r "{BASE_DIR / "requirements-analisis.txt"}"',
                    "install_packages_command": f'"{sys.executable}" -m pip install {paquetes}',
                }
            ), 500

        inspection = inspect_video(video_path)
        manifest, logs = run_real_analysis(video_path, metronome_path, output_dir)

        return jsonify(
            {
                "status": "success",
                "mensaje": "Video procesado correctamente con el análisis real.",
                "archivo": video_path.name,
                "ruta_video": str(video_path.resolve()),
                "ruta_audio_metronomo": str(metronome_path.resolve()),
                "tamano_bytes": inspection["tamano_bytes"],
                "tamano_mb": inspection["tamano_mb"],
                "metadata_extraida": manifest.get("metadata") or extract_metadata_from_filename(nombre_archivo or video_path.name),
                "analisis": {
                    "motor": "analyze_video_colab_refactor",
                    "video": inspection["video"],
                    "output_dir": str(output_dir.resolve()),
                    "stdout_preview": logs["stdout"][-4000:],
                    "stderr_preview": logs["stderr"][-4000:],
                },
                "generated_files": manifest.get("generated_files", []),
                "graficos": manifest.get("generated_plots", []),
            }
        )
    except FileNotFoundError as error:
        return jsonify({"status": "error", "message": str(error)}), 404
    except RuntimeError as error:
        return jsonify({"status": "error", "message": str(error)}), 500
    except Exception as error:
        return jsonify({"status": "error", "message": f"Error interno del análisis: {error}"}), 500


if __name__ == "__main__":
    import os
    print("Servidor de Python activo")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
