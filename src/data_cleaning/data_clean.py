import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data/raw/Housing.csv')
df.info()
df.head()
df.duplicated()


#missing val
print(df.isnull().sum())
round((df.isnull().sum() / df.shape[0]) * 100, 2)

df.duplicated()


#datatypes
num_col = [col for col in df.columns if df[col].dtype != 'object']
cat_col = [col for col in df.columns if df[col].dtype == 'object']
print('Num columns:', num_col)
print('cat_col:', cat_col)

#unique 
df[cat_col].nunique()
df[num_col].nunique()
