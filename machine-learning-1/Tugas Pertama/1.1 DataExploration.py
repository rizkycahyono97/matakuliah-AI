import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file
df = pd.read_csv('housing.csv')

# Display the first rows
print(df.head())

# summary the row
print(df.info())

# count missing values
missing_values = df.isnull().sum()
print(missing_values)

# statistik dasar untuk data numeric
print(df.describe())

# mendeteksi outlier menggunakan boxplot
plt.figure(figsize=(10,6))
sns.boxplot(data=df[[
  'median_house_value', 
  'median_income', 
  'housing_median_age'
  ]])
plt.title('Outlier dalam Fitur Numerik')
plt.savefig('/home/ch4rl0tt3/Documents/Code/Artifical Inteligent/outliers_plot.png')
