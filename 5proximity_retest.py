import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

# CONFIG
WAVE1_CSV = "apr8analysispipeline/may5wave1_ground_truth.csv"
WAVE2_CSV = "apr8analysispipeline/may5wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
OUTPUT_CSV = "apr8analysispipeline/retest_trial/proximity_output.csv"

# Columns
proximity_cols = [
    "PSYCHPROX_1_1", "PSYCHPROX_2_1", "PSYCHPROX_3_1", "PSYCHPROX_4_1",
    "PSYCHPROX_5_1", "PSYCHPROX_6_1", "PSYCHPROX_7_1", "PSYCHPROX_8_1"
]

# Reverse coding function
def reverse_proximity(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return 101 - numeric

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize email
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Apply reverse coding and compute normalized composites
def compute_proximity_composite(df, label):
    for col in proximity_cols:
        if col in df.columns:
            df[col] = reverse_proximity(df[col])
    df[f"Proximity_Composite_{label}"] = df[proximity_cols].mean(axis=1) / 100
    return df[["Email", f"Proximity_Composite_{label}"]]

df_w1 = compute_proximity_composite(df_w1, "W1")
df_w2 = compute_proximity_composite(df_w2, "W2")
df_pred = compute_proximity_composite(df_pred, "Pred")

# Merge all
merged_composite = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")

# Save
merged_composite.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")
