# Penerjemahan Mesin Saraf (NMT) Inggris-Indonesia dengan PyTorch

Proyek ini mengimplementasikan dan membandingkan dua arsitektur *Neural Machine Translation* (NMT) modern untuk tugas penerjemahan dari bahasa Inggris ke bahasa Indonesia. Arsitektur yang diimplementasikan adalah model *sequence-to-sequence* berbasis RNN dengan *Attention* sebagai *baseline*, dan model Transformer.

## Fitur

* **Preprocessing Data**: Termasuk pembersihan data, tokenisasi sub-kata menggunakan Byte-Pair Encoding (BPE), dan pembagian dataset menjadi set latih, validasi, dan uji.
* **Model Baseline**: Implementasi arsitektur Encoder-Decoder dengan GRU dan mekanisme atensi Bahdanau.
* **Model Transformer**: Implementasi arsitektur Transformer "Attention Is All You Need" menggunakan modul bawaan PyTorch.
* **Pelatihan & Evaluasi**: Skrip untuk melatih kedua model dari awal dan mengevaluasi performanya menggunakan metrik SacreBLEU.
* **Studi Ablasi**: Eksperimen untuk menganalisis dampak kedalaman (jumlah layer) pada performa model Transformer.
* **Notebook Lengkap**: Semua kode disajikan dalam satu file Jupyter Notebook (`.ipynb`) untuk kemudahan reproduksi.

## Dataset

Proyek ini menggunakan dataset bilingual Inggris-Indonesia yang bersumber dari [ManyThings.org (Anki)](https://www.manythings.org/anki/). Dataset ini berisi sekitar 14.881 pasang kalimat paralel.

## Persyaratan

Proyek ini dibuat dengan Python 3. Semua dependensi yang dibutuhkan tercantum dalam file `requirements.txt`.

* `torch`
* `tokenizers`
* `scikit-learn`
* `tqdm`
* `sacrebleu`
* `numpy`

## Instalasi

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/URL-REPO-ANDA/nama-repo.git](https://github.com/URL-REPO-ANDA/nama-repo.git)
    cd nama-repo
    ```

2.  **Buat Virtual Environment (Direkomendasikan)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/macOS
    # venv\Scripts\activate    # Untuk Windows
    ```

3.  **Instal Dependensi**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unduh Dataset**
    * Unduh dataset `ind.txt` dari [tautan ini](https://www.manythings.org/anki/ind-eng.zip).
    * Ekstrak dan letakkan file `ind.txt` di direktori utama proyek.

## Cara Penggunaan

Semua langkah, mulai dari preprocessing hingga evaluasi, terdapat di dalam Jupyter Notebook `machine-translation-nlp-pytorch.ipynb`. Buka notebook ini dan jalankan sel-sel kode secara berurutan.

Berikut adalah alur kerja utama di dalam notebook:

1.  **Tahap 1: Persiapan Data**
    * Sel-sel di bagian awal akan memuat file `ind.txt`.
    * Melatih tokenizer BPE untuk bahasa Inggris dan Indonesia, menyimpannya di direktori `tokenizers/`.
    * Membagi dataset menjadi file latih, validasi, dan uji, menyimpannya di direktori `data_split/`.

2.  **Tahap 2: Model Baseline (RNN + Attention)**
    * Definisi arsitektur model Encoder, Decoder, Attention, dan Seq2Seq.
    * Menjalankan *training loop* untuk model baseline. Model terbaik akan disimpan sebagai `baseline-rnn-model.pt`.
    * Evaluasi model baseline pada *test set* untuk mendapatkan skor BLEU dan melihat contoh terjemahan.

3.  **Tahap 3: Model Transformer**
    * Definisi arsitektur model Transformer.
    * Menjalankan *training loop* untuk model Transformer. Model terbaik akan disimpan sebagai `transformer-model.pt`.
    * Evaluasi model Transformer untuk mendapatkan skor BLEU dan membandingkannya dengan *baseline*.

4.  **Tahap 4: Studi Ablasi**
    * Definisi arsitektur Transformer yang dimodifikasi (dengan 1 layer).
    * Melatih dan mengevaluasi model yang dimodifikasi untuk menganalisis dampak perubahan arsitektur.

## Hasil

Hasil eksperimen menunjukkan bahwa model Transformer mencapai *validation loss* yang lebih rendah dibandingkan model RNN, mengindikasikan potensi belajar yang lebih baik. Namun, karena waktu pelatihan yang terbatas (10-15 epoch), kedua model belum sepenuhnya konvergen dan menghasilkan skor BLEU yang sangat rendah.

| Model             | Konfigurasi      | Val. Loss | Val. PPL | BLEU Score |
| ----------------- | ---------------- | :-------: | :------: | :--------: |
| RNN + Attention   | Baseline         |   4.482   |  88.44   |    0.07    |
| Transformer       | 3 Layers         | **4.213** | **67.59**|    0.01    |
| Transformer       | 1 Layer (Ablasi) |   4.458   |  86.28   |    0.01    |

Studi ablasi menunjukkan bahwa mengurangi kedalaman model Transformer dari 3 layer menjadi 1 layer meningkatkan *validation loss*, yang mengonfirmasi pentingnya arsitektur yang dalam untuk tugas ini.

Untuk detail lebih lanjut, silakan merujuk ke laporan `laporan-penelitian.docx`.

