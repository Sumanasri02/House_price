import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import joblib

def model_evaluation():
    model_path = "models/best_modle.pkl"
    test_data = "data/processed/test.csv"
    
    model = joblib.load(model_path)
    df = pd.read_csv(test_data)
    print(df.head())
    
    x = df.drop(columns="price", axis=1)
    y = df["price"]
    
    y_pred = model.predict(x)
    print("R2_score:", r2_score(y, y_pred))
    
if __name__ == "__main__":
    model_evaluation()