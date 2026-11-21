"""Sistem Informasi Klasifikasi Kerapian Sandal Laboratorium (PSD).
Backend Flask - Model XGBoost (14 Fitur: Warna RGB + GLCM Haralick).
Laboratorium Komputer - Pengolahan Sinyal & Citra Digital.
"""
import base64
import datetime
import os
import sys
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Dynamic path resolution
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import features as ft

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # Maksimal 32 MB

# Load Model Joblib
MODEL_FILE = BASE_DIR / "model_xgboost_kerapian.joblib"
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"File model tidak ditemukan di: {MODEL_FILE}")

print(f"[*] Memuat model dari: {MODEL_FILE}")
bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
transformer = bundle["transformer"]
label_encoder = bundle["label_encoder"]
feature_names = bundle["feature_names"]
classes = bundle["classes"]
print(f"[+] Model XGBoost berhasil dimuat! Kelas: {classes}")

DATASET_DIR = BASE_DIR / "foto-psd"


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def get_recommendation(label, confidence):
    """Menghasilkan pesan rekomendasi operasional berdasarkan hasil klasifikasi."""
    if label.lower() == "rapi":
        if confidence >= 85:
            return (
                "Kondisi penataan sandal sangat rapi dan sejajar. "
                "Area selasar masuk laboratorium memenuhi standar ketertiban operasional."
            )
        else:
            return (
                "Kondisi penataan sandal tergolong rapi, namun beberapa posisi sandal "
                "dapat dirapikan kembali agar lebih seragam."
            )
    else:
        if confidence >= 85:
            return (
                "PERINGATAN: Sandal berserakan dan tidak beraturan di area lantai. "
                "Harap segera mengingatkan praktikan atau merapikan sandal ke rak yang tersedia."
            )
        else:
            return (
                "Kondisi sandal terindikasi berantakan atau bertumpuk. "
                "Disarankan untuk menata ulang sebelum sesi laboratorium berikutnya."
            )


def run_prediction_pipeline(img_bgr, filename="citra_input"):
    """Pipeline pemrosesan citra, ekstraksi 14 fitur, transformasi, dan inferensi XGBoost."""
    if img_bgr is None or img_bgr.size == 0:
        return {"error": "Format citra tidak valid atau citra kosong."}

    # 1. Ekstraksi 14 fitur & tahapan visual
    try:
        feat_dict, stages = ft.extract_14_features_pro(img_bgr, return_stages=True)
    except Exception as e:
        return {"error": f"Gagal mengekstrak fitur: {str(e)}"}

    # 2. Susun dataframe sesuai urutan fitur saat pelatihan
    df_single = pd.DataFrame([feat_dict])[feature_names]

    # 3. Transformasi fitur dengan PowerTransformer yang sudah dilatih
    X_scaled = transformer.transform(df_single)

    # 4. Prediksi kelas dan probabilitas
    pred_idx = int(model.predict(X_scaled)[0])
    pred_proba = model.predict_proba(X_scaled)[0]

    predicted_label = str(classes[pred_idx])
    confidence = float(pred_proba[pred_idx] * 100)

    # Format dictionary probabilitas tiap kelas
    prob_dict = {cls: round(float(prob * 100), 2) for cls, prob in zip(classes, pred_proba)}

    # Detail 14 fitur dengan informasi grup dan deskripsi
    feature_details = []
    for f_name in feature_names:
        info = ft.FEATURE_INFO.get(f_name, {"desc": "-", "group": "Lainnya"})
        feature_details.append({
            "name": f_name,
            "value": round(float(feat_dict[f_name]), 4),
            "group": info["group"],
            "desc": info["desc"]
        })

    # Thumbnail asli yang diunggah
    # Resize thumbnail jika terlalu besar agar payload ringan
    h, w = img_bgr.shape[:2]
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        thumb_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        thumb_bgr = img_bgr
    original_b64 = ft.mat_to_base64_jpg(thumb_bgr)

    recommendation = get_recommendation(predicted_label, confidence)

    return {
        "success": True,
        "filename": filename,
        "label": predicted_label,
        "label_display": predicted_label.upper(),
        "confidence": round(confidence, 2),
        "probabilities": prob_dict,
        "features": feature_details,
        "stages": stages,
        "original_image_b64": original_b64,
        "recommendation": recommendation,
        "timestamp": datetime.datetime.now().strftime("%d %b %Y, %H:%M:%S")
    }


@app.route("/")
def index():
    """Halaman antarmuka utama sistem informasi."""
    return render_template(
        "index.html",
        model_name="XGBoost Classifier",
        model_accuracy="97.44%",
        feature_count=len(feature_names),
        classes=classes
    )


@app.route("/predict_file", methods=["POST"])
def predict_file():
    """Endpoint untuk upload file citra (JPG, PNG, WebP)."""
    if "file" not in request.files:
        return jsonify(error="Tidak ada file yang diunggah"), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify(error="Nama file kosong"), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify(error="File yang diunggah bukan format gambar yang didukung"), 400

        result = run_prediction_pipeline(img_bgr, filename=file.filename)
        if "error" in result:
            return jsonify(error=result["error"]), 400

        return jsonify(result)
    except Exception as e:
        return jsonify(error=f"Terjadi kesalahan saat memproses gambar: {str(e)}"), 500


@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    """Endpoint untuk tangkapan citra langsung dari kamera/webcam (Base64 data URL)."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "image" not in data:
            return jsonify(error="Data frame kamera tidak ditemukan"), 400

        image_data = data["image"]
        # Hilangkan header base64 jika ada (e.g., 'data:image/jpeg;base64,')
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify(error="Gagal mendekode tangkapan citra kamera"), 400

        result = run_prediction_pipeline(img_bgr, filename="Tangkapan_Kamera.jpg")
        if "error" in result:
            return jsonify(error=result["error"]), 400

        return jsonify(result)
    except Exception as e:
        return jsonify(error=f"Terjadi kesalahan saat memproses citra kamera: {str(e)}"), 500


@app.route("/predict_sample", methods=["POST"])
def predict_sample():
    """Endpoint untuk menguji citra sampel yang tersedia di server."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "category" not in data or "filename" not in data:
            return jsonify(error="Parameter category atau filename tidak lengkap"), 400

        category = data["category"]
        filename = data["filename"]
        target_path = DATASET_DIR / category / filename

        if not target_path.exists():
            return jsonify(error=f"File sampel '{filename}' tidak ditemukan di '{category}'"), 404

        img_bgr = cv2.imread(str(target_path))
        if img_bgr is None:
            return jsonify(error="Gagal membaca file citra sampel"), 500

        result = run_prediction_pipeline(img_bgr, filename=f"Sampel: {category}/{filename}")
        if "error" in result:
            return jsonify(error=result["error"]), 400

        return jsonify(result)
    except Exception as e:
        return jsonify(error=f"Terjadi kesalahan: {str(e)}"), 500


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Mengambil daftar 8 contoh gambar dari dataset (4 rapi, 4 berantakan)."""
    samples = []
    for cat in ["rapi", "berantakan"]:
        cat_dir = DATASET_DIR / cat
        if cat_dir.exists():
            valid_files = [
                f for f in sorted(os.listdir(cat_dir))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            # Ambil maksimal 4 sampel per kategori
            for fname in valid_files[:4]:
                samples.append({
                    "category": cat,
                    "filename": fname,
                    "url": f"/sample_file/{cat}/{fname}"
                })
    return jsonify(samples=samples)


@app.route("/sample_file/<category>/<filename>")
def serve_sample_file(category, filename):
    """Menyajikan file citra sampel untuk ditampilkan di UI."""
    target_dir = DATASET_DIR / category
    if not target_dir.exists():
        return jsonify(error="Kategori tidak ditemukan"), 404
    return send_from_directory(target_dir, filename)


@app.route("/api/info")
def api_info():
    """Informasi metadata sistem dan model."""
    return jsonify({
        "system_name": "Sistem Informasi Klasifikasi Kerapian Sandal Laboratorium Komputer",
        "model": "XGBoost Classifier (CRISP-DM)",
        "accuracy": "97.44%",
        "features": feature_names,
        "classes": classes,
        "status": "ready"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n" + "=" * 60)
    print(f"🚀 SISTEM INFORMASI KLASIFIKASI KERAPIAN LAB BERJALAN!")
    print(f"📍 Buka browser Anda di: http://localhost:{port}")
    print(f"=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
