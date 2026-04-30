#1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# 2. LOAD DATA
df = pd.read_csv(r"C:\Users\HP\Downloads\archive\creditcard.csv").sample(20000, random_state=42)

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# 3. DATA UNDERSTANDING
print("\nFraud vs Normal Count:\n")
print(df['Class'].value_counts())

# 4. VISUALIZATION
# Fraud vs Normal
plt.figure()
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

# Amount Distribution
plt.figure()
sns.histplot(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

# 5. FEATURE ENGINEERING
# Create Hour feature
df['Hour'] = df['Time'] // 3600

plt.figure()
sns.lineplot(x='Hour', y='Class', data=df)
plt.title("Fraud Transactions by Hour")
plt.show()

# 6.PREPARE DATA
X = df.drop('Class', axis=1)
y = df['Class']

# 7. HANDLE IMBALANCED DATA
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)

print("\nAfter SMOTE:\n")
print(pd.Series(y_res).value_counts())

# 8. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# 9. MODEL TRAINING
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 10. PREDICTION
y_pred = model.predict(X_test)

# 11. EVALUATION
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 12. RISK SCORE SYSTEM
y_prob = model.predict_proba(X_test)[:, 1]
risk_score = y_prob * 100

# 13. ALERT SYSTEM
def alert_system(prob):
    if prob > 0.8:
        return "HIGH RISK"
    elif prob > 0.5:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

alerts = [alert_system(p) for p in y_prob]

# Show sample results
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Risk Score (%)": risk_score,
    "Alert": alerts
})

print("\nSample Predictions:\n")
print(results.head())

# 14. FEATURE IMPORTANCE
importances = model.feature_importances_

plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance")
plt.show()

# 15. CUSTOMER SEGMENTATION
df['Amount_Group'] = pd.cut(df['Amount'], bins=5)

segmentation = df.groupby('Amount_Group', observed=False)['Class'].mean()
print("\nFraud Rate by Amount Group:\n")
print(segmentation)

# 16. SAVE MODEL
import joblib
joblib.dump(model, "fraud_model.pkl")

print("\nModel saved successfully!")
