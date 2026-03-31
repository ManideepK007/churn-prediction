import pytest
import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("src/model.joblib")
    sample_input = {
        "tenure": 0.1,  # Scaled example values, scale accordingly or retrain model to accept raw values with scaler included
        "MonthlyCharges": 0.05,
        "TotalCharges": 0.02,
        "Contract_Two year": 1,
        "InternetService_Fiber optic": 0,
        # Add other features as necessary per your one-hot encoding
    }
    df = pd.DataFrame([sample_input])
    pred = model.predict(df)
    assert pred[0] in [0, 1]
