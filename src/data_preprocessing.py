import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Drop customerID as it is an identifier, not useful for modeling
    df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric and remove rows with missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode target variable: Yes=1, No=0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Label encode binary categorical columns
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot encode remaining categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Scale numerical columns
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
