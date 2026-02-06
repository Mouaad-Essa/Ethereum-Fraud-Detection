import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load Data and Models
df = pd.read_csv('../fake_data_gen/real_batch_test.csv')
model = joblib.load('../models/my_best_fraud_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

print(f"Loaded {len(df)} rows of synthetic data.")

# 2. Separate the "Test" (X) from the "Answers" (y)
y_true = df['FLAG'] # The Answer Key
X_fake = df.drop(columns=['FLAG']) # The Exam Questions

# 3. PREPROCESS (Apply Log + Scale just like before)
columns_to_log = X_fake.columns # We used all columns in the generator

for c in columns_to_log:
    # Log transform safely
    X_fake[c] = X_fake[c].apply(lambda x: np.log(x) if x > 0 else 0)

# Scale
X_fake_scaled = scaler.transform(X_fake)

# 4. PREDICT
print("Running predictions...")
y_pred = model.predict(X_fake_scaled)

# 5. CALCULATE SCORE
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n--- FINAL REPORT CARD ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)
print("Confusion Matrix:")
print(f"Correctly identified Normal: {cm[0][0]} (out of 50)")
print(f"Correctly identified Fraud:  {cm[1][1]} (out of 50)")
print(f"False Alarms (Normal flagged as Fraud): {cm[0][1]}")
print(f"Missed Fraud (Fraud called Normal):     {cm[1][0]}")