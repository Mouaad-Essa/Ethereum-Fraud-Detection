import pandas as pd
import numpy as np

# 1. Load the original BIG dataset
# Make sure this path points to your actual file
source_file = '../../Data/address_data_combined_ts.csv'
df = pd.read_csv(source_file)

print(f"Loaded source data: {len(df)} rows found.")

# 2. Extract 50 REAL Normal Users (FLAG = 0)
real_normal = df[df['FLAG'] == 0].sample(n=50, random_state=42)

# 3. Extract 50 REAL Fraudsters (FLAG = 1)
real_fraud = df[df['FLAG'] == 1].sample(n=50, random_state=42)

# 4. Combine them into one test file
df_test = pd.concat([real_normal, real_fraud])

# 5. Clean up (Remove extra columns not needed for testing)
# We keep 'FLAG' because we need the answer key
cols_to_keep = [
    'FLAG',
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

# Select only the columns we need
df_test = df_test[cols_to_keep]

# 6. Save to CSV
df_test.to_csv('real_batch_test.csv', index=False)
print("✅ Created 'real_batch_test.csv' with 100 REAL rows (50 Normal, 50 Fraud)")