import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "apr8analysispipeline/may5wave1_ground_truth.csv"
WAVE2_CSV = "apr8analysispipeline/may5wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
OUTPUT_CSV = "apr8analysispipeline/retest_trial/ecdc_output.csv"

# Likert mapping for 5-point scale
likert_map_5pt = {
    "Strongly disagree": 1,
    "Somewhat agree": 2,
    "Neither agree nor disagree": 3,
    "Somewhat disagree": 4,
    "Strongly agree": 5
}

# Columns
indiv_cols = [f"ECDC_Individual_{i}" for i in range(1, 9)]
collect_cols = [f"ECDC_Collective_{i}" for i in range(1, 9)]

# Recode direction (unused here, but kept for future use)
def recode_direction(val):
    if val in [4, 5]:
        return "Agree"
    elif val in [1, 2]:
        return "Disagree"
    elif val == 3:
        return "Neutral"
    return np.nan

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize email
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Apply Likert mapping
for df in [df_w1, df_w2, df_pred]:
    for col in indiv_cols + collect_cols:
        if col in df.columns:
            df[col] = df[col].map(likert_map_5pt).astype(float)

# Compute composites
def compute_composites(df, prefix):
    df[f"{prefix}_Indiv"] = df[indiv_cols].mean(axis=1)
    df[f"{prefix}_Collect"] = df[collect_cols].mean(axis=1)
    return df[["Email", f"{prefix}_Indiv", f"{prefix}_Collect"]]

df_w1 = compute_composites(df_w1, "W1_Truth")
df_w2 = compute_composites(df_w2, "W2_Truth")
df_pred = compute_composites(df_pred, "Pred")

# Normalize to [0, 1]
for df in [df_w1, df_w2, df_pred]:
    for col in df.columns:
        if col != "Email":
            df[col] /= 5

# Merge all
merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")

# Save output
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")
