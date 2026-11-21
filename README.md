# Sistem Informasi Klasifikasi Kerapian Sandal Laboratorium Komputer

Aplikasi web berbasis Flask dan Artificial Intelligence untuk mendeteksi serta mengklasifikasi tingkat kerapian penataan sandal di depan laboratorium komputer secara otomatis, cepat, dan akurat. Sistem ini dikembangkan sebagai bagian dari proyek mata kuliah **Pengolahan Citra Digital (PSD)** dengan metodologi **CRISP-DM**.

---

## 🌟 Fitur Utama Aplikasi Web
1. **Klasifikasi via Upload Citra Lokal**:
   - Mendukung drag & drop maupun pemilihan file gambar (`JPG`, `PNG`, `WEBP`).
   - Preview gambar sebelum klasifikasi.
2. **Klasifikasi via Kamera / Webcam Langsung**:
   - Streaming kamera langsung di browser via WebRTC (`getUserMedia`).
   - Mode snapshot 1-klik (*Ambil & Klasifikasi*).
   - Fitur *Ganti Kamera* (mendukung kamera depan & belakang pada smartphone/tablet).
   - Mode *Auto-Scan*: memindai dan memprediksi secara otomatis setiap interval 4 detik.
3. **Galeri Sampel Lab 1-Klik**:
   - Tersedia galeri contoh citra dari kelas `rapi` dan `berantakan` yang bisa langsung diuji coba tanpa perlu mencari file foto di komputer.
4. **Hasil Analisis Komprehensif & Transparan**:
   - Status badge besar: **RAPI** (Hijau) atau **BERANTAKAN** (Merah).
   - Metrik *Confidence Score* dan persentase probabilitas tiap kelas.
   - Rekomendasi tindakan operasional untuk petugas/asisten lab.
   - Visualisasi 4 tahapan *preprocessing citra* (Citra Asli -> Center-Crop -> Penajaman Tepi Sobel -> Input GLCM 32 Level).
   - Rincian tabel nilai dari **14 Fitur Citra** (9 Momen Warna RGB + 5 GLCM Haralick).
5. **Riwayat Pemindaian (Session Log)**:
   - Mencatat hasil pemindaian selama sesi berjalan dengan thumbnail, label, dan waktu.

---

## 🏗️ Struktur File Aplikasi di Folder `Kerapian_Lab`
```
D:\kuliah\MatkulSem5\PSD\Kerapian_Lab\
├── app.py                             # Server backend Flask & routing API
├── features.py                        # Modul ekstraksi 14 fitur (Warna + GLCM)
├── model_xgboost_kerapian.joblib      # Model terlatih XGBoost & artefak preprocessing
├── dataset_fitur_sandal_14fitur.csv   # Dataset 14 fitur hasil augmentasi (582 sampel)
├── foto-psd/                          # Direktori citra dataset asli (rapi & berantakan)
│   ├── rapi/
│   └── berantakan/
├── templates/
│   └── index.html                     # Halaman antarmuka web modern (Tailwind CSS)
├── static/                            # Aset statis pendukung
├── run_web.bat                        # Script 1-klik menjalankan web di Windows
├── requirements.txt                   # Daftar dependensi library Python
└── README_WEB.md                      # Dokumentasi sistem web
```

---

## 🚀 Cara Menjalankan Aplikasi Web

### Cara 1: Menggunakan Script 1-Klik (Windows)
Cukup klik ganda (double-click) pada file:
```
run_web.bat
```
Browser akan otomatis membuka alamat `http://localhost:5000`.

### Cara 2: Melalui Terminal / Command Prompt
Buka terminal di folder `Kerapian_Lab`, lalu jalankan:
```bash
python app.py
```
Akses web melalui browser di:
```
http://localhost:5000
```

---

## 🔬 Spesifikasi Model & Ekstraksi Fitur
- **Algoritma Klasifikasi**: Extreme Gradient Boosting (**XGBoost Classifier**)
- **Akurasi Pengujian**: **97.44%**
- **Normalisasi Fitur**: PowerTransformer (Metode *Yeo-Johnson*)
- **Total Fitur**: Tepat **14 Fitur Citra**:
  - **9 Fitur Warna RGB**:
    - Mean Kanal Merah, Hijau, Biru (`Mean_R`, `Mean_G`, `Mean_B`)
    - Standar Deviasi Kanal Merah, Hijau, Biru (`Std_R`, `Std_G`, `Std_B`)
    - Skewness Kanal Merah, Hijau, Biru (`Skew_R`, `Skew_G`, `Skew_B`)
  - **5 Fitur Tekstur Spasial GLCM (Haralick)**:
    - `Energy`, `Entropy`, `Contrast`, `Correlation`, `Homogeneity`
    - Dihitung setelah *center-crop* 12%-88%, CLAHE, penajaman tepi Sobel, dan kuantisasi derajat keabuan 32 level pada multi-jarak $d=[1, 3, 5]$ dan 4 sudut orientasi.
