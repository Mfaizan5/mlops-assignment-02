import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Encode categorical columns automatically
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].astype("category").cat.codes

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model trained and saved successfully")
