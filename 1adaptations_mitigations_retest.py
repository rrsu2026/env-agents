import pandas as pd
import numpy as np

# Config
WAVE1_CSV = "apr8analysispipeline/may5wave1_ground_truth.csv"
WAVE2_CSV = "apr8analysispipeline/may5wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
OUTPUT_CSV = "apr8analysispipeline/retest_trial/adaptation_mitigation_output.csv"

# Mappings
likert_map_7pt = {
    "Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3,
    "Unsure": 4, "Somewhat agree": 5, "Agree": 6, "Strongly agree": 7
}

def reverse_code(series, max_value):
    return max_value + 1 - series

# Column groups
adaptation_cols = ["Adaptation_1", "Adaptation_2", "Adaptation_3"]
mitigation_cols = [
    "Mitigation_1", "Mitigation_2", "Mitigation_3", "Mitigation_4", "Mitigation_5",
    "Mitigation_6", "Mitigation_7", "Mitigation_8", "Mitigation_9", "Mitigation_10", "Mitigation_11"
]
reverse_mitigation = ["Mitigation_2", "Mitigation_4", "Mitigation_7", "Mitigation_9"]

def load_and_preprocess(csv_path, reverse_cols):
    df = pd.read_csv(csv_path)
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()
    for col in adaptation_cols + mitigation_cols:
        df[col] = df[col].map(likert_map_7pt).astype(float)
    for col in reverse_cols:
        df[col] = reverse_code(df[col], 7)
    return df

# Load data
df_w1 = load_and_preprocess(WAVE1_CSV, reverse_mitigation)
df_w2 = load_and_preprocess(WAVE2_CSV, reverse_mitigation)
df_pred = load_and_preprocess(PREDICTED_CSV, reverse_mitigation)

# compute composites
def compute_composites(df, prefix):
    df[f"{prefix}_Adaptation"] = df[adaptation_cols].mean(axis=1)
    df[f"{prefix}_Mitigation"] = df[mitigation_cols].mean(axis=1)
    df[f"{prefix}_Overall"] = df[[f"{prefix}_Adaptation", f"{prefix}_Mitigation"]].mean(axis=1)
    return df[["Email", f"{prefix}_Adaptation", f"{prefix}_Mitigation", f"{prefix}_Overall"]]

df_w1 = compute_composites(df_w1, "Wave1_Truth")
df_w2 = compute_composites(df_w2, "Wave2_Truth")
df_pred = compute_composites(df_pred, "Pred")

# merge on email
merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")

# normalize
for col in merged.columns:
    if col != "Email":
        merged[col] = merged[col] / 7.0

# save as csv
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")
