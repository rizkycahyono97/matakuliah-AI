# Klasifikasi Judul Berita (Olahraga vs. Politik) dengan RNN

Proyek ini merupakan implementasi model *deep learning* untuk klasifikasi teks biner, yang bertujuan membedakan judul berita berbahasa Indonesia antara kategori **Olahraga** dan **Politik**. Model ini dibangun menggunakan arsitektur *Recurrent Neural Network* (RNN), khususnya **Bidirectional LSTM (Bi-LSTM)**, dengan framework TensorFlow/Keras.

Proyek ini dibuat untuk memenuhi tugas mata kuliah [Nama Mata Kuliah Anda], dengan penekanan pada proses eksplorasi, iterasi, dan analisis kritis dalam pengembangan model.

## ✨ Fitur Utama

  - **Arsitektur RNN**: Menggunakan *stacked Bidirectional LSTM* untuk pemahaman konteks teks dari dua arah.
  - **Preparasi Data Cerdas**: Menerapkan metode **deteksi kata kunci (keyword detection)** untuk melabeli data secara otomatis dari judul berita mentah, karena label kategori tidak tersedia secara eksplisit.
  - **Eksperimen Terstruktur**: Kode dirancang dengan sel konfigurasi terpusat untuk memfasilitasi eksperimen dengan *hyperparameter* yang berbeda.
  - **Evaluasi Komprehensif**: Menganalisis performa model menggunakan metrik akurasi, *loss*, serta visualisasi *learning curve* dan *confusion matrix*.
  - **Kode Bersih & Modular**: Kode disajikan dalam notebook Jupyter dengan penjelasan di setiap langkahnya.

## 📊 Dataset

  - **Sumber**: Dataset yang digunakan adalah **[Indonesian News Dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/faizahmp/indonesian-news-dataset)** dari platform Kaggle.
  - **Metode Preparasi**:
    1.  Dataset asli yang berisi ribuan berita dibaca sebagai file CSV tunggal.
    2.  Karena kolom `source` tidak cukup spesifik untuk pelabelan, dikembangkan strategi baru dengan mendefinisikan daftar kata kunci untuk kategori 'sport' dan 'politik'.
    3.  Setiap judul berita diklasifikasikan berdasarkan keberadaan kata kunci tersebut.
    4.  Sebanyak **100 sampel acak** diambil dari setiap kategori yang berhasil diidentifikasi untuk membentuk dataset akhir yang seimbang (total 200 data).

## 📂 Struktur Proyek

```
.
├── klasifikasi_berita_rnn.ipynb  # File Notebook utama berisi seluruh kode
├── Laporan_Tugas.pdf             # Laporan analisis mendalam dan refleksi
└── README.md                     # File ini
```

*Catatan: Dataset tidak disertakan dalam repositori ini dan harus diunduh dari Kaggle atau di-attach langsung di lingkungan notebook Kaggle.*

## 🚀 Instalasi

Untuk menjalankan proyek ini di lingkungan lokal, ikuti langkah-langkah berikut.

**1. Prasyarat**

  - Python 3.8 atau lebih tinggi
  - `pip` package manager
  - Git

**2. Clone Repositori**

```bash
git clone https://github.com/rizkycahyono97/matakuliah-AI
cd  machine-learning-2/assesment-week-05
```

**3. Instalasi Dependensi**
Sangat disarankan untuk membuat dan mengaktifkan *virtual environment* sebelum menginstal library.

```bash
# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv
source venv/bin/activate  # Di Windows: venv\Scripts\activate

# Instal semua library yang dibutuhkan
pip install tensorflow scikit-learn matplotlib pandas numpy jupyterlab
```

## ▶️ Cara Menjalankan

Proyek ini dijalankan melalui file Jupyter Notebook (`.ipynb`).

1.  **Selesaikan Proses Instalasi**: Pastikan semua library pada langkah sebelumnya telah terinstal.
2.  **Siapkan Dataset**: Jika berjalan di Kaggle, pastikan dataset telah ditambahkan ke notebook. Jika berjalan lokal, unduh dataset dan sesuaikan path pada **Sel 2 (Konfigurasi)**.
3.  **Buka Jupyter**: Jalankan perintah `jupyter lab` atau `jupyter notebook` di terminal Anda.
4.  **Jalankan Notebook**:
      - Buka file `klasifikasi_berita_rnn.ipynb`.
      - Jalankan setiap sel kode secara berurutan dari atas ke bawah.
      - **Perhatian Khusus pada Sel 3**: Pada sel ini, Anda dapat melihat dan menyesuaikan daftar `sport_keywords` dan `politik_keywords` jika ingin bereksperimen dengan kata kunci yang berbeda.

## 📈 Hasil

Model final (Bi-LSTM 2 lapis) berhasil mencapai **akurasi validasi puncak sebesar 95.0%**. Hasil ini menunjukkan bahwa model mampu mempelajari pola dari judul berita dan menggeneralisasikannya dengan baik pada data yang belum pernah dilihat.

Analisis lebih detail mengenai performa, grafik *learning curve*, dan *confusion matrix* dapat ditemukan di dalam notebook dan `Laporan_Tugas.pdf`.

## 🛠️ Teknologi yang Digunakan

  - Python
  - TensorFlow & Keras
  - Scikit-learn
  - Pandas & NumPy
  - Matplotlib
  - Jupyter Notebook