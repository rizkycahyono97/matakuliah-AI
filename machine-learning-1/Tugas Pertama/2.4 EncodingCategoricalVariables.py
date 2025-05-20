import pandas as pd
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
print(df_encoded.head())