import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/Telco-Customer-Churn.csv')
print(df.head())
print(df['Churn'].value_counts())

sns.countplot(x='Churn', data=df)
plt.show()
