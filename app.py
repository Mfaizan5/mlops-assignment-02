import joblib
import pandas as pd

# Load model
model_path = r"models\model.pkl"
model = joblib.load(model_path)

# Prepare data exactly like the training data
data = pd.DataFrame([{
    "assignment_text": "Example assignment text",
    "course_name": "AI101",
    "gold_label": 0,
    "pred_label": 0,
    "pred_prob": 0.0
    # add all other columns used during training
}])

# Run prediction
prediction = model.predict(data)
print("Prediction:", prediction)
