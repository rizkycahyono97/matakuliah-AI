import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file
df = pd.read_csv("housing.csv")

# bloxplot sebelum handling outlier
plt.figure(figsize=(10, 6))
sns.boxenplot(data=df[['median_house_value', 'median_income']])
plt.title('sebelum menangani outlier')
plt.savefig('/home/ch4rl0tt3/Documents/Code/Artifical Inteligent/Tugas Pertama/2.2Sebelum_outlier.png')

# Menerapkan capping untuk outlier
def cap_outliers(df, column):
  q1 = df[column].quantile(0.05)
  q3 = df[column].quantile(0.95)
  df[column] = np.clip(df[column], q1, q3)

cap_outliers(df, 'median_income')
cap_outliers(df, 'median_house_value')

# bloxplot setelah handling outlier
plt.figure(figsize=(10, 6))
sns.boxenplot(data=df[['median_house_value', 'median_income']])
plt.title('setelah menangani outlier')
plt.savefig('/home/ch4rl0tt3/Documents/Code/Artifical Inteligent/Tugas Pertama/2.2Setelah_outlier.png')
