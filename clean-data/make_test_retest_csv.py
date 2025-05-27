"""
Cleans a raw Qualtrics export to produce a structured ground truth CSV file
for downstream analysis. The script removes metadata rows, retains only 
completed responses, identifies duplicate participants by email, and 
separates them into Wave 1 and Wave 2 based on submission date.
"""

import pandas as pd

# CONFIGURATION
RAW_CSV_PATH = 'apr8analysispipeline/may5rawgroundtruth.csv'
CLEANED_CSV_PATH = 'apr8analysispipeline/may5cleangroundtruth.csv'
WAVE1_PATH = 'apr8analysispipeline/may5wave1_ground_truth.csv'
WAVE2_PATH = 'apr8analysispipeline/may5wave2_ground_truth.csv'

# LOAD RAW FILE
df_raw = pd.read_csv(RAW_CSV_PATH, header=None)
new_header = df_raw.iloc[0]
df_clean = df_raw.iloc[2:].copy()
df_clean.columns = new_header

# STANDARDIZE AND FORMAT
df_clean["Email"] = df_clean["Email"].astype(str).str.lower().str.strip()
df_clean["Progress"] = pd.to_numeric(df_clean["Progress"], errors='coerce')
df_clean["EndDate"] = pd.to_datetime(df_clean["EndDate"], errors='coerce')

# KEEP ONLY COMPLETED RESPONSES
df_clean = df_clean[df_clean["Progress"] == 100]

# IDENTIFY DUPLICATES
duplicates = df_clean[df_clean.duplicated("Email", keep=False)].copy()

# SPLIT INTO WAVES
wave1_rows = []
wave2_rows = []

for email, group in duplicates.groupby("Email"):
    if len(group) == 2:
        sorted_group = group.sort_values("EndDate")
        wave1_rows.append(sorted_group.iloc[0])
        wave2_rows.append(sorted_group.iloc[1])
    else:
        print(f"⚠️ Email {email} has {len(group)} duplicates; skipping")

# TRIM TO RELEVANT COLUMNS AFTER SPLITTING
start_idx = df_clean.columns.get_loc("NEPS_1")
end_idx = df_clean.columns.get_loc("SS_Q6_11")
question_cols = df_clean.columns[start_idx:end_idx + 1].tolist()
final_cols = ["Email"] + question_cols

# SAVE CLEANED FULL DATA
df_clean[final_cols].to_csv(CLEANED_CSV_PATH, index=False)
print(f"\n✅ Cleaned ground truth CSV saved to: {CLEANED_CSV_PATH}")

# SAVE WAVES (with trimmed columns and Email first)
pd.DataFrame(wave1_rows)[final_cols].to_csv(WAVE1_PATH, index=False)
pd.DataFrame(wave2_rows)[final_cols].to_csv(WAVE2_PATH, index=False)
print(f"📥 Wave 1 CSV saved to: {WAVE1_PATH}")
print(f"📤 Wave 2 CSV saved to: {WAVE2_PATH}")

# LOG QUESTION COUNT
print(f"\n📊 Question range from NEPS_1 to SS_Q6_11: {len(question_cols)} questions.")
