import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# Load data
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Features and target
X = df.drop(["Churn",'customerID'], axis=1,errors='ignore')
y = df["Churn"]

# Define categorical and numeric features
categorical_features = ['gender', 'Contract', 'Dependents', 'DeviceProtection']
numeric_features = ['tenure', 'MonthlyCharges', 'SeniorCitizen']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ])

# Full pipeline: preprocess + classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data before training/tuning
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Hyperparameter grid for tuning
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best parameters found:", grid_search.best_params_)
print("Best ROC AUC:", grid_search.best_score_)

# Save the best model pipeline (preprocessing + tuned model)
joblib.dump(grid_search.best_estimator_, 'src/best_model_pipeline.joblib')
print("Best model pipeline saved!")
