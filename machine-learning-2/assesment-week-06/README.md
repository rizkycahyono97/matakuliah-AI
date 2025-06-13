# Eksplorasi Convolutional Autoencoder pada Fashion MNIST

Proyek ini merupakan implementasi dan eksplorasi model *deep learning* **Convolutional Autoencoder (CAE)**. Tujuan utamanya adalah untuk memahami bagaimana model belajar merepresentasikan data gambar dalam ruang berdimensi rendah (*latent space*) dan kemudian merekonstruksinya kembali. Proyek ini menggunakan dataset **Fashion MNIST** sebagai studi kasus dan dikembangkan menggunakan TensorFlow/Keras.

Selain rekonstruksi dasar, proyek ini juga mengeksplorasi aplikasi kreatif seperti **Denoising Autoencoder** dan **Visualisasi Ruang Laten**.

*Contoh hasil: Baris atas adalah gambar asli, baris bawah adalah hasil rekonstruksi oleh model.*

## ✨ Fitur Utama

  - **Arsitektur Simetris**: Implementasi Convolutional Autoencoder (CAE) dengan arsitektur Encoder dan Decoder yang simetris untuk rekonstruksi gambar yang akurat.
  - **Reduksi Dimensi**: Model mampu memadatkan gambar 28x28 piksel menjadi representasi laten yang jauh lebih kecil.
  - **Analisis Ruang Laten**: Kemampuan untuk memvisualisasikan ruang laten dan menunjukkan bagaimana model secara *unsupervised* mengelompokkan kategori pakaian yang serupa.
  - **Denoising Autoencoder**: Eksplorasi kemampuan model untuk membersihkan gambar dari *noise* acak, membuktikan bahwa ia mempelajari fitur yang robust.
  - **Kode Modular**: Dibangun menggunakan Keras Functional API untuk memisahkan logika Encoder, Decoder, dan Autoencoder secara jelas.

## 📊 Dataset

  - **Nama**: **Fashion MNIST**
  - **Sumber**: Dataset ini dimuat langsung menggunakan fungsi bawaan Keras `tf.keras.datasets.fashion_mnist.load_data()`. Tidak perlu mengunduh file CSV atau biner secara manual.
  - **Deskripsi**: Terdiri dari 70.000 gambar grayscale berukuran 28x28 piksel, terbagi dalam 10 kategori pakaian (T-shirt, Trouser, Sneaker, dll.).

## 📂 Struktur Proyek

```
.
├── autoencoder_fashion_mnist.ipynb  # File Notebook utama berisi seluruh kode
├── Laporan_Tugas.pdf                # Laporan analisis mendalam dan refleksi
└── README.md                        # File ini
```

## 🚀 Instalasi

Untuk menjalankan proyek ini di lingkungan lokal, ikuti langkah-langkah berikut.

**1. Prasyarat**

  - Python 3.8 atau lebih tinggi
  - `pip` package manager
  - Git

**2. Clone Repositori**

```bash
git clone https://github.com/rizkycahyono97/matakuliah-AI
cd machine-learning-2/assesment-week-06
```

**3. Instalasi Dependensi**
Sangat disarankan untuk membuat dan mengaktifkan *virtual environment* sebelum menginstal library.

```bash
# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv
source venv/bin/activate  # Di Windows: venv\Scripts\activate

# Instal semua library yang dibutuhkan
pip install tensorflow numpy matplotlib jupyterlab
```

*Catatan: Proyek ini sangat direkomendasikan untuk dijalankan di Google Colab atau Kaggle untuk memanfaatkan akselerasi GPU gratis.*

## ▶️ Cara Menjalankan

Proyek ini dijalankan sepenuhnya melalui file Jupyter Notebook.

1.  **Selesaikan Proses Instalasi**: Pastikan semua library pada langkah sebelumnya telah terinstal.
2.  **Buka Jupyter**: Jalankan perintah `jupyter lab` atau `jupyter notebook` di terminal Anda.
3.  **Jalankan Notebook**:
      - Buka file `autoencoder_fashion_mnist.ipynb`.
      - Jalankan setiap sel kode secara berurutan dari atas ke bawah.
      - Alur notebook mencakup: Persiapan Data -\> Pembangunan Model -\> Pelatihan -\> Evaluasi Rekonstruksi -\> Visualisasi Ruang Laten.
4.  **Eksperimen**:
      - Untuk melakukan eksplorasi, Anda dapat mengubah parameter di **Sel 2 (Konfigurasi)**.
      - Cobalah mengubah nilai `LATENT_DIM` menjadi `2` atau `8` untuk melihat bagaimana hal tersebut memengaruhi kualitas rekonstruksi dan sebaran titik pada visualisasi ruang laten.

## 📈 Hasil

  - **Rekonstruksi Gambar**: Model berhasil merekonstruksi bentuk dan struktur umum dari setiap item pakaian, meskipun dengan sedikit kehilangan detail halus (cenderung *blurry*), yang merupakan karakteristik dari kompresi autoencoder.
  - **Visualisasi Ruang Laten**: Model mampu mengelompokkan item pakaian yang sejenis ke dalam gugusan (*cluster*) yang berdekatan di ruang laten, membuktikan bahwa ia telah mempelajari fitur visual yang bermakna tanpa diberi label (secara *unsupervised*).
  - **Denoising**: Eksperimen tambahan menunjukkan model juga mampu bertindak sebagai *denoiser* yang efektif.

Detail visual, grafik, dan analisis mendalam dapat ditemukan di dalam notebook dan file `Laporan_Tugas.pdf`.

## 🛠️ Teknologi yang Digunakan

  - Python
  - TensorFlow / Keras
  - NumPy
  - Matplotlib
  - Google Colab