"""Modul Ekstraksi Fitur Citra Kerapian Sandal Laboratorium (PSD).
Mengekstrak tepat 14 fitur (9 Momen Warna RGB + 5 Tekstur GLCM Haralick)
sesuai metodologi CRISP-DM pada proyek Pengolahan Citra Digital.
"""
import base64
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

# Nama 14 fitur yang sesuai dengan model XGBoost
FEATURE_NAMES = [
    'Mean_R', 'Mean_G', 'Mean_B',
    'Std_R', 'Std_G', 'Std_B',
    'Skew_R', 'Skew_G', 'Skew_B',
    'Energy', 'Entropy', 'Contrast', 'Correlation', 'Homogeneity'
]

# Keterangan fitur untuk tampilan UI yang informatif
FEATURE_INFO = {
    'Mean_R': {'desc': 'Rata-rata intensitas kanal Merah (Red)', 'group': 'Warna'},
    'Mean_G': {'desc': 'Rata-rata intensitas kanal Hijau (Green)', 'group': 'Warna'},
    'Mean_B': {'desc': 'Rata-rata intensitas kanal Biru (Blue)', 'group': 'Warna'},
    'Std_R': {'desc': 'Standar deviasi intensitas kanal Merah', 'group': 'Warna'},
    'Std_G': {'desc': 'Standar deviasi intensitas kanal Hijau', 'group': 'Warna'},
    'Std_B': {'desc': 'Standar deviasi intensitas kanal Biru', 'group': 'Warna'},
    'Skew_R': {'desc': 'Kemencengan (Skewness) kanal Merah', 'group': 'Warna'},
    'Skew_G': {'desc': 'Kemencengan (Skewness) kanal Hijau', 'group': 'Warna'},
    'Skew_B': {'desc': 'Kemencengan (Skewness) kanal Biru', 'group': 'Warna'},
    'Energy': {'desc': 'Uniformitas/keseragaman tekstur spasial GLCM', 'group': 'Tekstur'},
    'Entropy': {'desc': 'Tingkat keacakan derajat keabuan (Shannon Entropy)', 'group': 'Tekstur'},
    'Contrast': {'desc': 'Variasi kontras lokal derajat keabuan sandal vs lantai', 'group': 'Tekstur'},
    'Correlation': {'desc': 'Korelasi linier ketergantungan piksel GLCM', 'group': 'Tekstur'},
    'Homogeneity': {'desc': 'Tingkat kedekatan distribusi GLCM terhadap diagonal', 'group': 'Tekstur'}
}


def mat_to_base64_jpg(img_bgr, quality=85):
    """Konversi citra OpenCV BGR ke base64 string data URL."""
    ret, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        return ""
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def extract_14_features_pro(img_bgr, return_stages=False):
    """Ekstraksi tepat 14 fitur dengan Edge-Enhanced GLCM & 9 Momen Warna RGB.

    Parameters:
        img_bgr (np.ndarray): Citra input dalam format BGR.
        return_stages (bool): Jika True, kembalikan juga citra tahapan preprocessing (base64).

    Returns:
        dict: 14 nilai fitur numerik.
        (opsional jika return_stages=True) dict: base64 citra tahapan preprocessing.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Citra kosong atau tidak valid.")

    # 1. Center Crop 12% - 88% untuk membuang area dinding/lantai kosong di tepi
    h, w = img_bgr.shape[:2]
    crop_ymin, crop_ymax = int(h * 0.12), int(h * 0.88)
    crop_xmin, crop_xmax = int(w * 0.12), int(w * 0.88)
    
    # Validasi dimensi setelah crop
    if crop_ymax <= crop_ymin or crop_xmax <= crop_xmin:
        img_crop = img_bgr
    else:
        img_crop = img_bgr[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Resize ke 256x256 untuk konsistensi spasial komputasi GLCM
    img_resized = cv2.resize(img_crop, (256, 256), interpolation=cv2.INTER_AREA)

    # Konversi ke RGB dan Grayscale
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # A. 9 FITUR WARNA (Mean, Std, Skewness tiap channel R, G, B)
    # ---------------------------------------------------------
    R = img_rgb[:, :, 0].flatten().astype(np.float64)
    G = img_rgb[:, :, 1].flatten().astype(np.float64)
    B = img_rgb[:, :, 2].flatten().astype(np.float64)

    mean_r, mean_g, mean_b = float(np.mean(R)), float(np.mean(G)), float(np.mean(B))
    std_r, std_g, std_b    = float(np.std(R)), float(np.std(G)), float(np.std(B))
    skew_r, skew_g, skew_b = float(skew(R)), float(skew(G)), float(skew(B))

    # ---------------------------------------------------------
    # B. PENAJAMAN TEPI & KONTRAS PADA GRAYSCALE SEBELUM GLCM
    # ---------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(img_gray)

    sobelx = cv2.Sobel(gray_clahe, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_clahe, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobelx, sobely)
    edge_gray = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Gabungkan 60% citra grayscale CLAHE + 40% ketajaman garis tepi sandal
    hybrid_gray = cv2.addWeighted(gray_clahe, 0.6, edge_gray, 0.4, 0)

    # Kuantisasi ke 32 level derajat keabuan (0-31)
    gray_quantized = (hybrid_gray // 8).astype(np.uint8)

    # ---------------------------------------------------------
    # C. 5 FITUR GLCM (Energy, Entropy, Contrast, Correlation, Homogeneity)
    # ---------------------------------------------------------
    angles_rad = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(
        gray_quantized,
        distances=[1, 3, 5],
        angles=angles_rad,
        levels=32,
        symmetric=True,
        normed=True
    )

    energy      = float(np.mean(graycoprops(glcm, 'energy')))
    contrast    = float(np.mean(graycoprops(glcm, 'contrast')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))

    # Shannon Entropy dari matriks GLCM
    entropy_list = []
    for d_idx in range(3):
        for a_idx in range(len(angles_rad)):
            P = glcm[:, :, d_idx, a_idx]
            ent = -np.sum(P * np.log2(P + 1e-10))
            entropy_list.append(ent)
    entropy = float(np.mean(entropy_list))

    features = {
        'Mean_R': mean_r, 'Mean_G': mean_g, 'Mean_B': mean_b,
        'Std_R': std_r,   'Std_G': std_g,   'Std_B': std_b,
        'Skew_R': skew_r, 'Skew_G': skew_g, 'Skew_B': skew_b,
        'Energy': energy,
        'Entropy': entropy,
        'Contrast': contrast,
        'Correlation': correlation,
        'Homogeneity': homogeneity
    }

    if return_stages:
        stages = {
            'crop_b64': mat_to_base64_jpg(img_resized),
            'gray_b64': mat_to_base64_jpg(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)),
            'edge_b64': mat_to_base64_jpg(cv2.cvtColor(edge_gray, cv2.COLOR_GRAY2BGR)),
            'hybrid_b64': mat_to_base64_jpg(cv2.cvtColor(hybrid_gray, cv2.COLOR_GRAY2BGR))
        }
        return features, stages

    return features
