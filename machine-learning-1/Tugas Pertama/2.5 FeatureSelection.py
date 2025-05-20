import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("housing.csv")

# Menerapkan Min-Max scaling
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[['median_income', 'housing_median_age', 'median_house_value']] = scaler.fit_transform(
    df[['median_income', 'housing_median_age', 'median_house_value']]
)
# One-Hot encoding untuk data kategorikal
df_encoded = pd.get_dummies(df_scaled, columns=['ocean_proximity'], drop_first=True)


# Matriks Korelasi
correlation_matrix = df_encoded.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi')
plt.savefig('/home/ch4rl0tt3/Documents/Code/Artifical Inteligent/Tugas Pertama/2.5MatriksKorelasi.png')

selected_features = df_encoded[['median_income', 'housing_median_age', 'total_rooms', 'ocean_proximity_NEAR BAY']]
print(selected_features.head())
