Tugas Kelompok - Implementasi Generative Model
📄 Deskripsi Proyek
Notebook ini berisi implementasi model Generative sederhana menggunakan dataset gambar (misalnya MNIST atau Fashion MNIST), untuk mengeksplorasi konsep pembelajaran tidak terawasi (unsupervised learning) dengan pendekatan autoencoder atau model generatif lainnya.

👥 Anggota Kelompok
- Rizky Cahyono Putra
- Raffa Arvel Nafi'Nadindra
- M. Irfansyah
- Galang Fachrezy
- Syaifan Nur Iwawan

🧠 Tujuan Pembelajaran
- Mempelajari konsep dasar dari autoencoder / generative model

- Memahami proses encoding dan decoding representasi laten

- Mengevaluasi kemampuan model dalam merekonstruksi data input

- Melakukan eksplorasi terhadap kualitas dan variasi hasil generatif

🛠️ Teknologi yang Digunakan
1. Python 3

2. NumPy

3. Matplotlib

4. PyTorch / TensorFlow (sesuai isi notebook)

5. Jupyter Notebook

▶️ Cara Menjalankan Proyek
1. Clone repositori atau unduh file notebook:

   git clone <URL_REPOSITORI>


2. Siapkan environment:
   Pastikan kamu sudah memiliki Python 3 dan package berikut:

   pip install numpy matplotlib torch torchvision notebook

3. Jalankan Jupyter Notebook:
  
   jupyter notebook

   Lalu buka file t08-tugas-kelompok-implementasi-generative-a.ipynb.

4. Langkah dalam Notebook:

   - Jalankan sel secara berurutan dari atas ke bawah.

   - Bagian awal akan melakukan import library dan persiapan dataset.

   - Selanjutnya model akan didefinisikan dan dilatih.

   - Terakhir, visualisasi output hasil rekonstruksi akan ditampilkan.

📁 Struktur Notebook
1. Import Library

2. Persiapan Dataset

3. Arsitektur Model

4. Training Model

5. Visualisasi Hasil Rekonstruksi

6. Eksperimen Representasi Laten

7. Analisis dan Kesimpulan

🧪 Hasil Eksperimen
- Model berhasil melakukan rekonstruksi gambar dengan akurasi visual yang cukup baik

- Representasi laten menunjukkan kemampuan mengkompres informasi penting dari input

- Eksplorasi interpolasi dalam ruang laten memberikan gambaran variasi output yang realistis

📌 Catatan Penting
- Dataset yang digunakan telah dinormalisasi antara 0 dan 1

- Model dilatih selama N epoch (disesuaikan)

- Latent space berukuran kecil untuk memudahkan visualisasi dan interpretasi


📚 Referensi
- Deep Learning with PyTorch - Official Docs

- Autoencoder Tutorial - Towards Data Science

- Dataset: MNIST / FashionMNIST