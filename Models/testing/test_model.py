import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = '../models/my_best_fraud_model.pkl'
SCALER_PATH = '../models/scaler.pkl'

# --- 2. LOAD SAVED FILES ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Error: Could not find model or scaler files. Check the 'models' folder.")
    exit()

print("Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Loaded successfully!")

# --- 3. REAL FRAUD DATA (Extracted from Dataset) ---
fake_data = {
    'Avg min between sent tnx': [1179.02],
    'Avg min between received tnx': [1124.89],
    'Time Diff between first and last (Mins)': [25126.45],
    'Unique Received From Addresses': [13],
    'min value received': [0.0],
    'max value received ': [0.75],
    'avg val received': [0.176666667],
    'min val sent': [0.145],
    'avg val sent': [0.419269824],
    'total transactions (including tnx to create contract': [22],
    'total ether received': [2.65],
    # Note: Negative balance is technically impossible in real Ethereum,
    # but likely due to a floating point error in the dataset.
    # The model sees "Negative/Zero" as very suspicious.
    'total ether balance': [-0.28488877],
    'adjusted_eth_value__absolute_sum_of_changes': [7.66977754],
    'adjusted_eth_value__mean_abs_change': [0.3652275019047619],
    'adjusted_eth_value__energy_ratio_by_chunks__num_segments_10__segment_focus_0': [0.0202370540854782],
    'adjusted_eth_value__sum_values': [-0.2848887700000001],
    'adjusted_eth_value__abs_energy': [3.335465711308085],
    'adjusted_eth_value__ratio_value_number_to_time_series_length': [0.4090909090909091],
    'adjusted_eth_value__quantile__q_0.1': [-0.267420588],
    'adjusted_eth_value__count_below__t_0': [0.3636363636363636],
    'adjusted_eth_value__count_above__t_0': [0.6818181818181818],
    'adjusted_eth_value__median': [0.15],
}

df_test = pd.DataFrame(fake_data)

# --- 4. PREPROCESS (Must match training EXACTLY) ---
columns_to_log = [
    'Avg min between sent tnx', 'Avg min between received tnx',
    'Time Diff between first and last (Mins)', 'Unique Received From Addresses',
    'min value received', 'max value received ', 'avg val received',
    'min val sent', 'avg val sent',
    'total transactions (including tnx to create contract',
    'total ether received', 'total ether balance',
    'adjusted_eth_value__absolute_sum_of_changes',
    'adjusted_eth_value__mean_abs_change',
    'adjusted_eth_value__energy_ratio_by_chunks__num_segments_10__segment_focus_0',
    'adjusted_eth_value__sum_values', 'adjusted_eth_value__abs_energy',
    'adjusted_eth_value__ratio_value_number_to_time_series_length',
    'adjusted_eth_value__quantile__q_0.1', 'adjusted_eth_value__count_below__t_0',
    'adjusted_eth_value__count_above__t_0', 'adjusted_eth_value__median'
]

# Apply Log Transform
for c in columns_to_log:
    if c in df_test.columns:
        # Avoid log(negative) or log(0) errors
        df_test[c] = df_test[c].apply(lambda x: np.log(x) if x > 0 else 0)

# Apply Scaler
df_test_scaled = scaler.transform(df_test)

# --- 5. PREDICT ---
print("\n--- 🤖 AI Verdict ---")
prediction = model.predict(df_test_scaled)
probability = model.predict_proba(df_test_scaled)

is_fraud = prediction[0] == 1
confidence = probability[0][1] if is_fraud else probability[0][0]

if is_fraud:
    print(f"ALERT: FRAUD DETECTED (Confidence: {confidence*100:.2f}%)")
else:
    print(f"CLEAN: Normal Transaction (Confidence: {confidence*100:.2f}%)")