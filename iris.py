import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')



path = kagglehub.dataset_download("arshid/iris-flower-dataset")

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "arshid/iris-flower-dataset/versions/1",
  "Iris.csv",
)

# print(df.head()) # 5 rows of data, we have length and width for speal and pedal + the species
# print(df.describe()) # no striking outliers or anomalies (e.g negative numbers)
# print(df.info()) # all columns are float64
# print(df.isnull().sum()) # no null values
# print(df.duplicated().sum()) # no duplicates

ax = sns.pairplot(df, hue='species') # visualizing the data
ax.fig.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()