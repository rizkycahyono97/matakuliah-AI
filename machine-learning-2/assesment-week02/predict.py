import torch
import pandas as pd
from model import IrisMLP
from sklearn.preprocessing import StandardScaler

# Load model
model = IrisMLP()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Contoh data baru
sample = [[5.1, 3.5, 1.4, 0.2]]  # Ubah sesuai kebutuhan
scaler = StandardScaler()
df = pd.read_csv('Iris.csv')
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
scaler.fit(X)
sample_scaled = scaler.transform(sample)

# Prediksi
sample_tensor = torch.FloatTensor(sample_scaled)
output = model(sample_tensor)
_, predicted_class = torch.max(output, 1)
classes = df['Species'].unique()
print("Prediksi kelas:", classes[predicted_class])
