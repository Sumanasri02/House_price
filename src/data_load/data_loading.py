import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/Housing.csv')
print(df)
df.info()
df.head()
df.duplicated()
df.describe()
df.shape()
