import pandas as pd
import joblib

# Load the best tuned model pipeline
model = joblib.load('src/best_model_pipeline.joblib')

# Load sample data or new data to predict on
df = pd.read_csv('data/Telco-Customer-Churn.csv')

# Preprocess target column if it exists (not needed for prediction, but useful to keep)
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop target and ID columns (assume customerID is present)
X = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')

# Predict churn probabilities
probs = model.predict_proba(X)[:, 1]  # Probability of churn = class 1

# Predict churn class (0 = stay, 1 = churn)
preds = model.predict(X)

# Show prediction for first customer as an example
print(f"Prediction for first customer: {'Churn' if preds[0] == 1 else 'Stay'} with probability {probs[0]:.2f}")

# Optional: save prediction results with probabilities to CSV
df_results = df.copy()
df_results['Churn_Prediction'] = preds
df_results['Churn_Probability'] = probs
df_results.to_csv('data/predictions.csv', index=False)

print("Predictions saved to data/predictions.csv")

