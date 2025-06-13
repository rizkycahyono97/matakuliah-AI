# Proyek Deteksi Penggunaan Masker dengan Transfer Learning (PyTorch)

Proyek ini adalah implementasi model *deep learning* untuk mengklasifikasikan gambar individu yang memakai masker dan tidak memakai masker. Proyek ini dibangun sebagai bagian dari tugas mata kuliah [Nama Mata Kuliah Anda], menggunakan metode **Transfer Learning** dengan arsitektur **ResNet18** praterlatih pada framework PyTorch.

## ✨ Fitur Utama

  - **Transfer Learning**: Menggunakan model ResNet18 yang sudah dilatih pada ImageNet untuk ekstraksi fitur yang kuat.
  - **Fine-Tuning**: Menerapkan teknik *layer freezing* dan mengganti *classifier head* agar sesuai dengan 2 kelas (memakai/tanpa masker).
  - **Dataset Kustom**: Menggunakan `torch.utils.data.Dataset` untuk memuat dataset gambar secara efisien.
  - **Validasi & Evaluasi**: Memisahkan data latih dan validasi (80/20) untuk evaluasi yang objektif.
  - **Visualisasi Hasil**: Menampilkan grafik akurasi & loss, serta *confusion matrix* untuk analisis performa model.
  - **Penyimpanan Model**: Menyimpan bobot model dengan performa validasi terbaik secara otomatis.

## 📂 Struktur Proyek

Berikut adalah struktur file dan direktori penting dalam proyek ini:

```
.
├── 2_dataset_pilihan_200/
│   ├── with_mask/
│   │   └── ... (100 gambar acak)
│   └── without_mask/
│       └── ... (100 gambar acak)
│
├── notebook_klasifikasi_masker.ipynb   # File utama berisi seluruh kode
├── best_mask_classifier.pth            # File output bobot model terbaik
├── Laporan_Proyek.pdf                  # Laporan detail proyek
└── README.md                           # File ini
```

## 📊 Dataset

Dataset yang digunakan adalah versi kurasi dari **[Face Mask \~12K Images Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)** yang tersedia di Kaggle.

Untuk memenuhi spesifikasi tugas, tidak semua gambar digunakan. Sebaliknya, dilakukan proses **pengambilan sampel acak** untuk memilih **100 gambar** dari setiap kelas. Proses ini memastikan dataset akhir (total 200 gambar) tetap memiliki variasi yang baik dan seimbang.

## 🚀 Instalasi

Untuk menjalankan proyek ini di lingkungan lokal, ikuti langkah-langkah berikut.

**1. Prasyarat**

  - Python 3.8 atau lebih tinggi
  - `pip` package manager
  - Git

**2. Clone Repositori**

```bash
git clone [URL_GITHUB_ANDA]
cd [NAMA_FOLDER_REPO_ANDA]
```

**3. Instalasi Dependensi**
Disarankan untuk menggunakan *virtual environment* agar tidak mengganggu instalasi Python global Anda.

```bash
# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv
source venv/bin/activate  # Di Windows: venv\Scripts\activate

# Instal semua library yang dibutuhkan
pip install torch torchvision scikit-learn matplotlib numpy tqdm Pillow jupyterlab
```

## ▶️ Cara Menjalankan

Proyek ini dirancang untuk dijalankan sebagai Jupyter Notebook.

**1. Pastikan Dataset Siap**
Pastikan direktori `2_dataset_pilihan_200/` sudah ada di dalam folder proyek dan berisi subfolder `with_mask` dan `without_mask`.

**2. Buka Jupyter Notebook**
Jalankan perintah berikut di terminal Anda:

```bash
jupyter lab
```

atau

```bash
jupyter notebook
```

Perintah ini akan membuka antarmuka Jupyter di browser Anda.

**3. Jalankan Sel Kode**

  - Buka file `notebook_klasifikasi_masker.ipynb`.
  - Jalankan setiap sel kode secara berurutan dari atas ke bawah. Alur notebook adalah sebagai berikut:
    1.  **Konfigurasi**: Mengatur semua parameter utama.
    2.  **Persiapan Dataset**: Memuat dan membagi data.
    3.  **Arsitektur Model**: Membangun model ResNet18.
    4.  **Proses Training**: Melatih model dan menyimpan bobot terbaik.
    5.  **Evaluasi & Visualisasi**: Menampilkan grafik dan *confusion matrix*.
    6.  **Prediksi Gambar Tunggal**: Menguji model pada satu gambar contoh.

## 📈 Hasil

Model yang dilatih berhasil mencapai **akurasi validasi puncak sebesar 97.5%**. Hasil ini menunjukkan efektivitas metode *transfer learning* untuk tugas klasifikasi gambar bahkan dengan dataset yang relatif kecil.

Analisis lebih mendalam mengenai hasil dan metrik performa dapat ditemukan di `Laporan_Proyek.pdf`.

## 🛠️ Teknologi yang Digunakan

  - **Python**
  - **PyTorch**
  - **Scikit-learn**
  - **Matplotlib**
  - **Jupyter Notebook**