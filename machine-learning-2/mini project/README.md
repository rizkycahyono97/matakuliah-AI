
# Klasifikasi Aksara Jawa Tulisan Tangan Menggunakan CNN

Proyek ini bertujuan untuk membangun model klasifikasi gambar menggunakan Convolutional Neural Network (CNN)  untuk mengenali aksara Jawa tulisan tangan. Model dilatih dari awal menggunakan framework TensorFlow, dengan data dari folder terstruktur yang berisi gambar aksara.

## Anggota Kelompok
- Rizky Cahyono Putra  
- Raffa Arvel Nafi'Nadindra  
- Syaifan Nur  
- Irfansyah  
- Muhammad Galang Fachrezy

# 📌 Fitur Utama
- Preprocessing gambar: konversi ke grayscale, resize, dan normalisasi.

- Arsitektur CNN sederhana.

- Pelatihan model menggunakan torchvision dan torch.

- Evaluasi akurasi pada data validasi dan uji.

- Visualisasi hasil prediksi dan akurasi pelatihan.

## 📁 Struktur Direktori


├── klasifikasi-aksara-jawa-tulisan-tangan-menggunakan.ipynb
├── /hanacaraka/
│   ├── ha/
│   ├── na/
│   ├── ca/
│   └── 


## ⚙️ Konfigurasi dan Setup

Simpan dataset yang telah di download dalam folder bernama 'hanacaraka' lalu nalankan notebook menggunakan lingkungan Python yang telah memiliki:

pip install tensorflow matplotlib numpy scikit-learn opencv-python


## 🚀 Cara Menjalankan

1. **Buka Notebook**
   Jalankan notebook di Jupyter atau Google Colab:

   bash
   jupyter notebook klasifikasi-aksara-jawa-tulisan-tangan-menggunakan.ipynb
   

2. **Struktur Data Otomatis**
   Dataset akan otomatis dibaca dengan perintah berikut:

   python
   train_ds = tf.keras.utils.image_dataset_from_directory(
       "/kaggle/input/hanacaraka",
       validation_split=0.2,
       subset="training",
       seed=123,
       image_size=(64, 64),
       batch_size=32,
       color_mode='grayscale'
   )
  

3. **Training Model**
   Model CNN dibangun dengan augmentasi dan LeakyReLU activation:

   python
   model = models.Sequential([
       layers.Input(shape=(64, 64, 1)),
       layers.RandomFlip("horizontal"),
       layers.RandomRotation(0.1),
       layers.RandomZoom(0.1),
       layers.Rescaling(1./255),

       layers.Conv2D(64, (3, 3), padding='same'),
       layers.LeakyReLU(0.2),
       layers.MaxPooling2D((2, 2)),
       layers.BatchNormalization(),

       layers.Conv2D(128, (3, 3), padding='same'),
       layers.LeakyReLU(0.2),
       layers.MaxPooling2D((2, 2)),
       layers.BatchNormalization(),

       layers.Conv2D(256, (3, 3), padding='same'),
       layers.LeakyReLU(0.2),
       layers.MaxPooling2D((2, 2)),

       layers.Flatten(),
       layers.Dense(256),
       layers.LeakyReLU(0.2),
       layers.Dropout(0.5),
       layers.Dense(NUM_CLASSES)  # output logits
   ])


4. **Evaluasi dan Visualisasi**

   - Contoh gambar dari dataset
   - Plot akurasi dan loss per epoch
   - Confusion matrix klasifikasi


## 📊 Hasil Model
- model mencapai akurasi validasi sebesar 95%
- Visualisasi `loss` dan `accuracy` disediakan otomatis.
- Confusion matrix membantu melihat kesalahan klasifikasi.

---

## ✅ Fitur Tambahan

- Augmentasi Data: Random flip, zoom, dan rotasi.
- Optimasi Dataset: Caching dan prefetching untuk efisiensi GPU.

## 📌 Catatan Penting

- Dataset harus sudah terstruktur di subfolder sesuai nama aksara.
- Input gambar diubah ke ukuran `(64, 64)` dan dalam format grayscale.
- Gunakan GPU (Colab/Kaggle) untuk training yang lebih cepat.

