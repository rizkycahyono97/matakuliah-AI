import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the file
df = pd.read_csv("housing.csv")

# Menerapkan Min-Max scaling
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[['median_income', 'housing_median_age', 'median_house_value']] = scaler.fit_transform(
    df[['median_income', 'housing_median_age', 'median_house_value']]
)

# Menampilkan statistik ringkasan sebelum dan sesudah scaling
print("\nSebelum Scaling:")
print(df[['median_income', 'housing_median_age', 'median_house_value']].describe())

print("\n\nSetelah Scaling:")
print(df_scaled[['median_income', 'housing_median_age', 'median_house_value']].describe())
