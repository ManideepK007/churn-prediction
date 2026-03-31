from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
import os
model_path = os.path.join(os.path.dirname(__file__), "..", "src", "model_pipeline.joblib")
model_pipeline = joblib.load(model_path)
@app.route("/")
def home():
    return jsonify({
        "message": "Hello from Flask!",
        "status": "Churn prediction API is running!"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        prediction = model_pipeline.predict(df)
        return jsonify({"churn_prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({
            "message": "Hello from Flask!",
            "status": "Churn prediction API is running!"
            })

if __name__ == "__main__":
    print("Flask app is starting...")
    print("Registered routes:")
    print(app.url_map)
    app.run(debug=True)
