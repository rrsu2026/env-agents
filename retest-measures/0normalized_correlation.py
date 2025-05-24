import os
import pandas as pd
from scipy.stats import pearsonr

# CONFIG
INPUT_DIR = "apr8analysispipeline/retest_trial"  
OUTPUT_CSV = "apr8analysispipeline/normalized_corr_trial/americanvoices_norm_corr.csv"

# Detect sub-composites by suffix patterns
def get_matching_columns(columns, suffix):
    return [col for col in columns if col.endswith(suffix)]

# Normalized correlation
def compute_normalized_r(w1, w2, pred):
    try:
        r_w1_w2 = pearsonr(w1, w2)[0]
        r_w1_pred = pearsonr(w1, pred)[0]
        return r_w1_pred / r_w1_w2 if r_w1_w2 != 0 else float('nan')
    except Exception:
        return float('nan')


results = []

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".csv"):
        continue

    path = os.path.join(INPUT_DIR, fname)
    df = pd.read_csv(path)
    cols = df.columns

    # shared composite or sub-composites
    w1_cols = get_matching_columns(cols, "_W1")
    w2_cols = get_matching_columns(cols, "_W2")
    pred_cols = get_matching_columns(cols, "_Pred")

    # only keys that exist in all 3
    shared_keys = set(col.replace("_W1", "") for col in w1_cols)
    shared_keys &= set(col.replace("_W2", "") for col in w2_cols)
    shared_keys &= set(col.replace("_Pred", "") for col in pred_cols)

    for key in sorted(shared_keys):
        col_w1 = f"{key}_W1"
        col_w2 = f"{key}_W2"
        col_pred = f"{key}_Pred"

        sub_df = df[[col_w1, col_w2, col_pred]].dropna()
        if len(sub_df) >= 3:  # min 3 rows for stable correlation
            norm_r = compute_normalized_r(sub_df[col_w1], sub_df[col_w2], sub_df[col_pred])
            results.append({
                "Scale": key,
                "File": fname,
                "N": len(sub_df),
                "Normalized_Correlation": round(norm_r, 3)
            })

# Save output
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved normalized correlations to: {OUTPUT_CSV}")
