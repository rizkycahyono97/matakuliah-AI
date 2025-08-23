# Klasifikasi Citra Menggunakan Model CNN

## Anggota Kelompok
- Rizky Cahyono Putra (rizkycahyonoputra80@student.cs.unida.gontor.ac.id)  
- M. Irfansyah (mirfansyah26@student.cs.unida.gontor.ac.id)  
- Raffa Arvel Nafi’Nadindra (raffaarvel@gmail.com)  
- Muhammad Galang Fachrezy (muhammadgalangfachrezy22@student.cs.unida.gontor.ac.id)  
- Syaifan Nur Iwawan (syaifannoeriwawan78@student.cs.unida.gontor.ac.id)  

## Abstrak
Klasifikasi citra merupakan salah satu permasalahan penting dalam bidang *computer vision*. Penelitian ini berfokus pada implementasi Convolutional Neural Network (CNN) dengan arsitektur klasik (ResNet-18/VGG16) untuk menyelesaikan permasalahan klasifikasi citra pada dataset **Intel Image Classification**. Dataset terdiri dari enam kelas citra pemandangan: *buildings, forest, glacier, mountain, sea, street* dengan total sekitar 25.000 gambar ukuran 150 × 150 piksel. Preprocessing dilakukan dengan resizing, normalisasi, serta data augmentation. Model dilatih menggunakan *categorical cross-entropy* dan Adam optimizer, mencapai akurasi **86%** pada data uji.  

**Kata kunci:** klasifikasi citra, CNN, deep learning, TensorFlow.

---

## I. Pendahuluan
Klasifikasi citra merupakan topik penting dalam *machine learning* dan *computer vision*. CNN menjadi pendekatan utama karena mampu mengekstraksi fitur spasial gambar secara hierarkis. Proyek ini menggunakan dataset **Intel Image Classification** (6 kelas: bangunan, hutan, gunung, laut, jalan, es) dengan tujuan membangun model CNN yang akurat.

## II. Metode
### A. Dataset
- Total ~25.000 gambar resolusi 150x150 px  
- Dibagi menjadi:
  - Train: 14.000 gambar  
  - Test: 3.000 gambar  
  - Predict: 7.000 gambar  
- Kelas: Buildings, Forest, Mountain, Sea, Street, Glacier  

### B. Arsitektur Model
CNN berbasis ResNet-18/VGG16, terdiri dari lapisan konvolusi (ReLU, BatchNorm, MaxPooling), fully connected, serta output *softmax*.  
Jika transfer learning digunakan, bobot awal pretrained ImageNet dengan fine-tuning.  

### C. Implementasi
Menggunakan **TensorFlow/Keras** di Kaggle.  
Tahapan preprocessing: resizing ke 150x150 px, normalisasi [0-1], RGB conversion, labeling, dan shuffling data.  

Model CNN terdiri dari:
- Conv2D (feature extraction)  
- MaxPooling2D (downsampling)  
- Flatten (konversi 2D ke 1D)  
- Dense (klasifikasi)  
- Dropout (mencegah overfitting)  
- Softmax (probabilitas kelas)  

Total parameter: ~6.8 juta.  

### D. Transfer Learning
Menggunakan **EfficientNetB0** pretrained ImageNet, layer dasar dibekukan, ditambah dense layer baru untuk 6 kelas.  

### E. Training
- Epochs = 20  
- Menggunakan validation data  
- *ReduceLROnPlateau* untuk adaptive learning rate  

## III. Hasil dan Analisis
### A. Grafik Akurasi dan Loss
Model konvergen dengan baik. Akurasi meningkat stabil, terdapat sedikit overfitting ringan. Loss training menurun konsisten, validation loss menurun dengan fluktuasi kecil.  

### B. Confusion Matrix
- Performa terbaik pada kelas *forest* dan *sea*.  
- Kesalahan sering terjadi antara *building vs street* dan *glacier vs mountain* karena karakteristik mirip.  

### C. Evaluasi
Model mencapai **86% akurasi** pada data uji.  

## IV. Kesimpulan
CNN berhasil diimplementasikan untuk klasifikasi citra pemandangan dengan Intel Image Classification dataset, menghasilkan akurasi **86%**. Meskipun terdapat kesalahan pada kelas dengan visual mirip, model tetap mampu melakukan generalisasi dengan baik.  

## Referensi
1. P. Bansal, “Intel Image Classification Dataset,” Kaggle, 2019. [Link](https://www.kaggle.com/puneet6060/intel-image-classification)
