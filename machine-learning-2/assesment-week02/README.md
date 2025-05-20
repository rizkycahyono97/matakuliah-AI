
# Klasifikasi Bunga Iris dengan Multilayer Perceptron (MLP)

## Deskripsi Proyek
Proyek ini merupakan implementasi algoritma Multilayer Perceptron (MLP) menggunakan PyTorch untuk klasifikasi bunga Iris berdasarkan dataset dari UCI Machine Learning Repository. Proyek ini bertujuan untuk:

- Mengimplementasikan model MLP dari awal menggunakan PyTorch
- Membandingkan performa MLP dengan model sederhana seperti Logistic Regression
- Melakukan visualisasi dan analisis performa model
- Menyusun laporan ilmiah berdasarkan eksperimen dan hasil

## Struktur Notebook
Notebook `laporan_akhir.ipynb` berisi:

1. **Eksplorasi Data**: Visualisasi data dan distribusi kelas
2. **Preprocessing**: Encoding label, normalisasi, dan split dataset
3. **Model MLP**: Arsitektur dengan 1 hidden layer dan fungsi aktivasi ReLU
4. **Training**: Optimasi menggunakan Adam dan CrossEntropyLoss selama 100 epoch
5. **Evaluasi**: Akurasi, confusion matrix, dan classification report
6. **Perbandingan**: Evaluasi Logistic Regression sebagai baseline
7. **Prediksi**: Prediksi data baru dengan model terlatih
8. **Analisis**: Pembahasan performa model dan aplikasinya di dunia nyata

## Arsitektur MLP
- Input Layer: 4 neuron (fitur Iris)
- Hidden Layer: 10 neuron, ReLU
- Output Layer: 3 neuron, klasifikasi multi-kelas (softmax output via CrossEntropyLoss)

## Tools
- Python 3.8+
- PyTorch
- Scikit-learn
- Pandas, Matplotlib, Seaborn

## Dataset
Dataset diambil dari: [https://www.kaggle.com/datasets/uciml/iris/data](https://www.kaggle.com/datasets/uciml/iris/data)

## Cara Menjalankan
1. Jalankan semua sel di `laporan_akhir.ipynb` menggunakan lingkungan seperti Kaggle Notebook.
2. Model akan dilatih selama 100 epoch dan hasil evaluasi akan ditampilkan.
3. Ubah input di bagian prediksi jika ingin mencoba data baru.

## Hasil
Model MLP menunjukkan performa akurasi tinggi dan mampu mengenali pola non-linear lebih baik daripada Logistic Regression, walaupun pada dataset sederhana seperti Iris, perbedaannya tidak terlalu signifikan.

## Tantangan di Dunia Nyata
- Overfitting pada data kecil
- Tuning hyperparameter
- Kebutuhan data besar untuk deep learning

## Tim
- Rizky Cahyono Putra
- Raffa Arvel
- Syaifan Nur
- Irfansyah
- Galang Alvian

## Lisensi
Proyek ini hanya digunakan untuk tujuan edukasi dalam rangka pemenuhan tugas mata kuliah Pembelajaran Mesin 2.
