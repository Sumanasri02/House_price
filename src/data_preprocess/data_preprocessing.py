import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def data_preprocess():

    df = pd.read_csv('data/raw/Housing.csv')

    numeric_df = df.select_dtypes(include=np.number)
    categorical_df = df.select_dtypes(exclude=np.number)

    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(numeric_df)
    scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_df.columns)

    encoded_categorical_df = pd.get_dummies(categorical_df, drop_first=True).astype(int)
    


    final_df = pd.concat([scaled_numeric_df, encoded_categorical_df], axis=1)

    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)

    # Save train and test CSV
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)



if __name__ == "__main__":
    data_preprocess()







