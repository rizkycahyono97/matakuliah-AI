# **Laporan Akhir: Convolutional Neural Network (CNN)**

## **Mata Kuliah**: Pembelajaran Mesin  
**Dosen Pengampu**: Dr. Oddy Virgantara Putra, S.Kom., M.T.  
**Kelompok**:
- Rizky Cahyono Putra
- Raffa Arvel Nafi’Nadindra
- Syaifan Nur
- Irfansyah
- Muhammad Galang Fachrezy  

---

## **Deskripsi Proyek**

Laporan ini menyajikan proyek pembelajaran mesin yang menggunakan **PyTorch** untuk membangun dan melatih model **Convolutional Neural Network (CNN)** dalam mengklasifikasikan angka dari dataset **MNIST**. Dataset MNIST adalah kumpulan data gambar grayscale berukuran 28x28 piksel yang berisi angka 0 hingga 9.

Proyek ini mencakup langkah-langkah utama dalam pipeline deep learning, mulai dari preprocessing data hingga evaluasi model. Berikut adalah penjelasan terstruktur dari setiap bagian:

---

## **Isi Laporan**

### **1. Import Library**
Library yang digunakan dalam proyek ini meliputi:
- **NumPy** dan **Pandas** untuk manipulasi data.
- **PyTorch** dan **torchvision.transforms** untuk membangun model deep learning.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
```

---

### **2. Dataset dan Preprocessing**
Dataset MNIST disimpan dalam format `.idx3-ubyte` untuk gambar dan `.idx1-ubyte` untuk label. Data dibaca menggunakan kelas kustom `MNISTDataset`, kemudian diproses menjadi tensor PyTorch.

- Fungsi membuka file biner dan memparsing header MNIST untuk mendapatkan informasi seperti jumlah data dan dimensi gambar.
- Gambar diubah ke tensor PyTorch untuk digunakan dalam pelatihan model.

---

### **3. Arsitektur Model CNN**
Model CNN dirancang dengan arsitektur berikut:
- **Tiga layer konvolusi** dengan padding untuk mempertahankan dimensi gambar.
- **Max pooling** untuk mengurangi dimensi spasial.
- **Layer fully connected** untuk mengonversi fitur spasial menjadi output klasifikasi.

Contoh arsitektur sederhana:
```python
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

### **4. Training Dataset**
Dataset MNIST dibagi menjadi **train_dataset** dan **test_dataset**:
- Gambar dan label diubah menjadi tensor PyTorch.
- DataLoader digunakan untuk membaca data dalam batch berukuran 8 dengan urutan yang diacak (**shuffle**) untuk meningkatkan generalisasi.

```python
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
```

---

### **5. Setup Perangkat & Model**
Kode memeriksa apakah GPU tersedia untuk mempercepat pelatihan model. Jika GPU tidak tersedia, model akan dijalankan pada CPU.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
```

---

### **6. Loss Function dan Optimizer**
- **Loss Function**: CrossEntropyLoss, cocok untuk klasifikasi multi-kelas.
- **Optimizer**: Adam, varian gradient descent yang efisien.
- Parameter tambahan:
  - Learning rate (`lr`) = 1e-4.
  - Weight decay = 1e-5 untuk mencegah overfitting.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

---

### **7. Parameter Model**
Menampilkan total parameter dan parameter yang dapat dilatih (**trainable**) dalam model.

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params}, Trainable Parameters: {trainable_params}")
```

---

### **8. Inisialisasi**
Model dilatih selama **5 epochs**, yang berarti model akan melewati seluruh dataset sebanyak 5 kali.

```python
num_epochs = 5
```

---

### **9. Evaluasi Model**
- Mode training diaktifkan menggunakan `model.train()`.
- Variabel `train_loss` digunakan untuk menghitung total loss selama 1 epoch.

```python
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")
```

---

### **10. Loop Data Testing**
Data testing diproses oleh model untuk mengevaluasi performa. Hasil prediksi dibandingkan dengan label asli untuk menghitung akurasi.

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
```

---

### **11. Menyimpan Model**
Model disimpan dalam format dictionary untuk digunakan kembali di masa mendatang.

```python
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss
}, 'cnn_mnist_model.pth')
```

---

### **12. Visualisasi dan Hasil**
Visualisasi hasil pelatihan dan pengujian disajikan untuk mempermudah analisis performa model.

---

### **13. Simpulan**
Model CNN yang dibuat berhasil mengenali angka dari dataset MNIST dengan akurasi tinggi. Pipeline lengkap, mulai dari preprocessing data hingga evaluasi akhir, telah berhasil diimplementasikan. Proyek ini memperkuat pemahaman praktis mengenai deep learning dan arsitektur CNN.

---

## **Cara Menjalankan Proyek**
1. Pastikan Anda telah menginstal library yang diperlukan:
   ```bash
   pip install torch torchvision numpy pandas
   ```
2. Unduh dataset MNIST dari sumber resmi atau gunakan dataset yang disediakan.
3. Jalankan kode Python untuk melatih model:
   ```bash
   python train_cnn.py
   ```
4. Evaluasi model menggunakan file testing.

---

## **Kontribusi**
Jika Anda ingin berkontribusi atau memiliki pertanyaan, silakan buat issue atau pull request.

--- 

Semoga repository ini membantu Anda memahami implementasi **Convolutional Neural Network (CNN)** menggunakan PyTorch! 🚀
