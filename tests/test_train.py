import os
import pandas as pd
import joblib

def test_dataset_exists():
    assert os.path.exists("data/dataset.csv")

def test_model_training():
    assert os.path.exists("models/model.pkl")

def test_data_loading():
    data = pd.read_csv("data/dataset.csv")
    assert not data.empty
