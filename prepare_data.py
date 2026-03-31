import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/Telco-Customer-Churn.csv')

# Convert 'Yes'/'No' churn to 1/0
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    if col != 'customerID':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
