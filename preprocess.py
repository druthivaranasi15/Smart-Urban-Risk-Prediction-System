import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("--- Starting BINARY Preprocessing (High Risk vs Low Risk) ---")

# 1. LOAD DATA
file_path = 'data/road_accidents.csv'
if not os.path.exists(file_path):
    print(f"ERROR: {file_path} not found.")
    exit()

df = pd.read_csv(file_path)
print(f"Dataset Loaded. Shape: {df.shape}")

# 2. DROP LEAKAGE COLUMNS
cols_to_drop = ['Number of Casualties', 'Number of Fatalities', 'Accident Location Details']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 3. BINARY TARGET ENCODING
# We combine 'Fatal' and 'Serious' into 'High Risk' (1)
# We keep 'Slight' as 'Low Risk' (0)
target_col = 'Accident Severity'

if target_col in df.columns:
    df = df.dropna(subset=[target_col])
    
    # Define the mapping logic
    def map_severity(val):
        val = str(val).lower()
        if 'fatal' in val or 'serious' in val:
            return 1 # High Risk
        else:
            return 0 # Low Risk (Slight)

    df['Target'] = df[target_col].apply(map_severity)
    
    print(f"Target Encoded as Binary. High Risk (1) vs Low Risk (0)")
    print(f"Class Distribution:\n{df['Target'].value_counts()}")
    
    df.drop(columns=[target_col], inplace=True)
else:
    print(f"ERROR: Target '{target_col}' not found.")
    exit()

# 4. TIME ENGINEERING
time_col = 'Time of Day'
if time_col in df.columns:
    try:
        temp_time = pd.to_datetime(df[time_col], format='%H:%M:%S', errors='coerce')
        if temp_time.notna().sum() > 0.5 * len(df):
            hour = temp_time.dt.hour.fillna(temp_time.dt.hour.mode()[0])
            df['Hour_Sin'] = np.sin(2 * np.pi * hour / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * hour / 24)
            df.drop(columns=[time_col], inplace=True)
    except:
        pass

# 5. CATEGORICAL ENCODING
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
for col in cat_cols:
    freq_map = df[col].value_counts(normalize=True).to_dict()
    df[col] = df[col].map(freq_map).fillna(0)

# 6. SAVE
df = df.fillna(0)
df.to_csv('data/processed_data.csv', index=False)
print("SUCCESS! Binary Processed data saved.")