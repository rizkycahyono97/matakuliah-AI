import pandas as pd

df = pd.read_csv("housing.csv")

# kolom dengan nilai yang hilang
missing_column = df.columns[df.isnull().any()]
print('Data yang hilang: ', missing_column)

# teknik 1: menghapus baris dengan nilai yang hilang
df_dropna = df.dropna();

# teknik 2: menginputasi nilai yang hilang
df_input = df.copy()
df_input['total_bedrooms'].fillna(df_input['total_bedrooms'].median(), inplace=True)

# menampilkan semua data 
print(df_input.isnull().sum())