import torch
import torch.nn as nn
import torch.optim as optim
from model import IrisMLP
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
df = pd.read_csv('Iris.csv')  # pastikan file ini tersedia
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = LabelEncoder().fit_transform(df['Species'])

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to Tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)

# Dataset dan DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model, loss, optimizer
model = IrisMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
EPOCHS = 100
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
    
    acc = correct / len(train_dataset)
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}')

# Save model
torch.save(model.state_dict(), 'model.pth')
