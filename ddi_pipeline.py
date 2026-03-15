# markdown Cell

# # 💊 DDI Risk Modeling + Polypharmacy Regimen Risk Score (RRS)
# **Dataset:** NSIDES (OffSIDES + TwoSIDES) — Tatonetti Lab  
# **Source:** https://tatonettilab-resources.s3.us-west-1.amazonaws.com/nsides/
# 
# ---
# 
# 
# ---
# 
# 



# markdown Cell

# ---
# ## 🔧 Phase 1 — Setup & Data Download



# Code Cell: code

# Cell 1.1 | Install dependencies
# imbalanced-learn : SMOTE for class imbalance handling
# xgboost          : Gradient boosting — best tabular ML performer for DDI data
# shap             : SHapley Additive exPlanations — model interpretability layer
# plotly           : Interactive visualizations for the RRS heatmap in demo

#!pip install imbalanced-learn xgboost shap plotly --quiet
print("Installation complete")



# Code Cell: code

import nbformat

def notebook_to_python_script(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    python_script_content = []
    for cell in notebook_content.cells:
        if cell.cell_type == 'code':
            # Add a comment to indicate the original cell type and ID
            python_script_content.append(f"\n# Code Cell: {cell.cell_type} (id: {cell.cell_id})\n")
            python_script_content.append(cell.source)
            python_script_content.append("\n") # Add a newline for separation
        elif cell.cell_type == 'markdown' or cell.cell_type == 'text':
            # Convert markdown to Python comments
            python_script_content.append(f"\n# Markdown Cell: {cell.cell_type} (id: {cell.cell_id})\n")
            for line in cell.source.split('\n'):
                python_script_content.append(f"# {line}")
            python_script_content.append("\n") # Add a newline for separation

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(python_script_content))

# Save the current notebook to a temporary .ipynb file
# This is a common Colab pattern to get the current notebook's content
#!jupyter nbconvert --to notebook --output-dir=/content --stdout /content/drive/MyDrive/Colab_Notebooks/ddi_risk_modeling.ipynb > /content/current_notebook.ipynb

# Convert the temporary notebook to a Python script
notebook_to_python_script('/content/current_notebook.ipynb', 'ddi_risk_modeling.py')

print("Notebook successfully converted to ddi_risk_modeling.py")



# Code Cell: code

from google.colab import files

# Download the generated Python script
files.download('ddi_risk_modeling.py')



# Code Cell: code

# Temporarily convert the notebook to a Python script and print its content
import nbformat

def notebook_to_python_script(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    python_script_content = []
    for cell in notebook_content.cells:
        if cell.cell_type == 'code':
            python_script_content.append(f"\n# Code Cell: {cell.cell_type} (id: {cell.cell_id})\n")
            python_script_content.append(cell.source)
            python_script_content.append("\n")
        elif cell.cell_type == 'markdown' or cell.cell_type == 'text':
            python_script_content.append(f"\n# Markdown Cell: {cell.cell_type} (id: {cell.cell_id})\n")
            for line in cell.source.split('\n'):
                python_script_content.append(f"# {line}")
            python_script_content.append("\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(python_script_content))

# NOTE: This command assumes the current notebook is saved to Google Drive at the specified path.
# If your notebook is saved elsewhere, please update the path accordingly.
# For this demonstration, we're relying on a previously saved notebook or a working `jupyter nbconvert` command.
# If `jupyter nbconvert` fails, the content might not be what's expected.

# Attempt to get the current notebook's content. This might still rely on Google Drive being mounted if the notebook is unsaved.
# For a more robust solution without Google Drive, the notebook itself would need to be saved to /content first.

# Create a dummy .ipynb file for conversion, as nbconvert often expects a file path.
# In a real scenario, you'd ensure the notebook is saved first.
#!jupyter nbconvert --to notebook --output-dir=/content --stdout '/content/drive/MyDrive/Colab_Notebooks/ddi_risk_modeling.ipynb' > /content/temp_current_notebook.ipynb

output_filename = 'ddi_risk_modeling.py'
notebook_to_python_script('/content/temp_current_notebook.ipynb', output_filename)

with open(output_filename, 'r', encoding='utf-8') as f:
    print(f.read())



# Code Cell: code

import nbformat
from google.colab import _message
import json # Added import for json

def notebook_to_python_script(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    python_script_content = []
    for cell in notebook_content.cells:
        if cell.cell_type == 'code':
            # Check if cell_id exists before adding it to the comment
            cell_id_str = f" (id: {cell.cell_id})" if hasattr(cell, 'cell_id') else ""
            python_script_content.append(f"\n# Code Cell: {cell.cell_type}{cell_id_str}\n")
            python_script_content.append(cell.source)
            python_script_content.append("\n")
        elif cell.cell_type == 'markdown' or cell.cell_type == 'text':
            # Markdown cells may not always have a cell_id, especially older ones.
            # We'll just use the cell_type for consistency.
            python_script_content.append(f"\n# {cell.cell_type} Cell\n")
            for line in cell.source.split('\n'):
                python_script_content.append(f"# {line}")
            python_script_content.append("\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(python_script_content))

# Request the content of the current notebook from the Colab frontend
# This is a Colab-specific way to get the *current* notebook's content reliably.
notebook_json_response = _message.blocking_request('get_ipynb', timeout_sec=120)

if notebook_json_response and 'ipynb' in notebook_json_response:
    # Convert the dict to a JSON string before writing
    notebook_content_str = json.dumps(notebook_json_response['ipynb'], indent=2)
    with open('/content/current_notebook_for_conversion.ipynb', 'w', encoding='utf-8') as f:
        f.write(notebook_content_str)
    print("Current notebook saved locally for conversion.")

    output_filename = 'ddi_risk_modeling.py'
    notebook_to_python_script('/content/current_notebook_for_conversion.ipynb', output_filename)

    with open(output_filename, 'r', encoding='utf-8') as f:
        print(f.read())
    print("\nNotebook content printed above.")
else:
    print("Failed to retrieve current notebook content. Please ensure the notebook is saved.")



# Code Cell: code

import nbformat
from google.colab import drive

def notebook_to_python_script(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    python_script_content = []
    for cell in notebook_content.cells:
        if cell.cell_type == 'code':
            python_script_content.append(f"\n# Code Cell: {cell.cell_type} (id: {cell.cell_id})\n")
            python_script_content.append(cell.source)
            python_script_content.append("\n")
        elif cell.cell_type == 'markdown' or cell.cell_type == 'text':
            python_script_content.append(f"\n# Markdown Cell: {cell.cell_type} (id: {cell.cell_id})\n")
            for line in cell.source.split('\n'):
                python_script_content.append(f"# {line}")
            python_script_content.append("\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(python_script_content))

# Mount Google Drive to access files from it
drive.mount('/content/drive')

# This command assumes the current notebook is saved to Google Drive at the specified path.
# If your notebook is saved elsewhere, please update the path accordingly.
# A robust alternative for a live notebook in Colab would be to use the Colab API if available,
# or to ensure the notebook is explicitly saved before this step.
#!jupyter nbconvert --to notebook --output-dir=/content --stdout '/content/drive/MyDrive/Colab_Notebooks/ddi_risk_modeling.ipynb' > /content/current_notebook.ipynb

# Convert the temporary notebook to a Python script
notebook_to_python_script('/content/current_notebook.ipynb', 'ddi_risk_modeling.py')

print("Notebook successfully converted to ddi_risk_modeling.py")



# Code Cell: code

from google.colab import files

# Download the generated Python script
files.download('ddi_risk_modeling.py')




# Code Cell: code

# Cell 1.2 | Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, gc, json, pickle
from itertools import combinations

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix, f1_score
)
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import shap

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

print(f"pandas  {pd.__version__}")
print(f"xgboost {xgb.__version__}")
print(f"shap    {shap.__version__}")
print("All imports OK")



# Code Cell: code

# Utility: load any NSIDES file robustly with auto column detection + numeric coercion
def load_nsides(filepath, col_map_fn, chunksize=None, nrows=None):
    """
    Reads a NSIDES csv.gz file, remaps columns to standard names,
    and coerces all numeric columns to float.

    col_map_fn : a function that takes a lowercased column name
                 and returns the standard name, or None to skip
    """
    header = pd.read_csv(filepath, compression='gzip', nrows=0)
    col_map = {}
    for col in header.columns:
        mapped = col_map_fn(col.strip().lower(), col)
        if mapped:
            col_map[col] = mapped

    NUM_COLS = ['A','B','C','D','PRR','PRR_error','mean_reporting_frequency']

    reader = pd.read_csv(
        filepath,
        compression='gzip',
        usecols=list(col_map.keys()),
        nrows=nrows,
        chunksize=chunksize
    )

    def process(df):
        df = df.rename(columns=col_map)
        for c in NUM_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    if chunksize:
        return (process(chunk) for chunk in reader)
    return process(reader)

print("load_nsides() utility defined")



# Code Cell: code

# Cell 1.3 | Download NSIDES datasets from Tatonetti Lab S3
# OffSIDES : single-drug off-label side effect signals (PSM-adjusted)
# TwoSIDES : drug-pair interaction signals (same PSM methodology)
#
# Column definitions (from README):
#   A = reports with drug [pair] AND side effect
#   B = reports with drug [pair] WITHOUT side effect
#   C = reports for PSM-matched controls WITH side effect
#   D = reports for PSM-matched controls WITHOUT side effect
#   PRR = (A/(A+B)) / (C/(C+D))  — signal strength
#   PRR_error = uncertainty estimate of PRR
#   mean_reporting_frequency = A/(A+B)

BASE = "https://tatonettilab-resources.s3.us-west-1.amazonaws.com/nsides/"
os.makedirs("data", exist_ok=True)
#
#!wget -q --show-progress -O data/OFFSIDES.csv.gz "{BASE}OFFSIDES.csv.gz"
#!wget -q --show-progress -O data/TWOSIDES.csv.gz "{BASE}TWOSIDES.csv.gz"
print("Downloads complete")



# markdown Cell

# ---
# ## 🔍 Phase 2 — Exploratory Data Analysis



# Code Cell: code

# Cell 2.1 | Load OffSIDES in chunks
# Why chunks? OffSIDES has ~3M rows.
# Chunk-based loading avoids OOM errors on Colab free tier (12 GB RAM).

OFFS_COLS = [
    'drug_rxnorn_id', 'drug_concept_name',
    'condition_meddra_id', 'condition_concept_name',
    'A', 'B', 'C', 'D', 'PRR', 'PRR_error', 'mean_reporting_frequency'
]

chunks = []
for chunk in pd.read_csv('data/OFFSIDES.csv.gz', chunksize=200_000,
                          usecols=OFFS_COLS, compression='gzip'):
    chunks.append(chunk)

offsides = pd.concat(chunks, ignore_index=True)
del chunks; gc.collect()

print(f"OffSIDES shape     : {offsides.shape}")
print(f"Unique drugs       : {offsides['drug_rxnorn_id'].nunique():,}")
print(f"Unique side effects: {offsides['condition_meddra_id'].nunique():,}")
offsides.head(3)



# Code Cell: code

# Force numeric types immediately after loading
for col in ['A','B','C','D','PRR','PRR_error','mean_reporting_frequency']:
    offsides[col] = pd.to_numeric(offsides[col], errors='coerce')



# Code Cell: code

# Cell 2.3 | Load TwoSIDES sample for EDA (1M rows)

TWO_COLS_ACTUAL = pd.read_csv('data/TWOSIDES.csv.gz', compression='gzip', nrows=0).columns.tolist()


twosides_sample = pd.read_csv(
    'data/TWOSIDES.csv.gz',
    nrows=1_000_000,
    compression='gzip'
)

# ── Force numeric types immediately ──────────────────────────────────────────
NUM_COLS = ['A', 'B', 'C', 'D', 'PRR', 'PRR_error', 'mean_reporting_frequency']
for col in NUM_COLS:
    twosides_sample[col] = pd.to_numeric(twosides_sample[col], errors='coerce')

print(f"\nSample shape : {twosides_sample.shape}")
print(f"PRR dtype    : {twosides_sample['PRR'].dtype}  ← must be float64")
print(f"Unique pairs : {twosides_sample.groupby(['drug_1_rxnorn_id','drug_2_rxnorm_id']).ngroups:,}")
twosides_sample.head(3)



# Code Cell: code

twosides_sample['PRR'].describe()



# Code Cell: code

for t in [2, 3, 5, 10]:
    print(f"Threshold {t}:",
          (twosides_sample["PRR"] >= t).sum())



# Code Cell: code

# Cell 2.5 | Top 20 side effects in high-PRR pairs(Proportional reporting ratio)
high_prr = twosides_sample[twosides_sample['PRR'] >= 5]
top_fx = (
    high_prr['condition_concept_name']
    .value_counts().head(20).reset_index()
)
top_fx.columns = ['Side Effect', 'Count']

plt.figure(figsize=(11, 6))
sns.barplot(data=top_fx, x='Count', y='Side Effect', palette='Reds_r')
plt.title('Top 20 Side Effects in High-PRR Drug Pairs (PRR >= 2)', fontsize=13)
plt.xlabel('Occurrence Count')
plt.tight_layout()
plt.savefig('top_effects.png', bbox_inches='tight')
plt.show()



# markdown Cell

# ---
# ## ⚙️ Phase 3 — Feature Engineering
# 
# ### Strategy (3-tier, per project guideline)
# 
# 1. **Drug-level features** from OffSIDES — each drug's baseline adverse event profile across all its side effects  
# 2. **Pair-level features** from TwoSIDES — the interaction signal for each drug pair across all shared side effects  
# 3. **Derived statistical features** — signal-to-noise ratio, log transforms, relative risk ratio (novel contribution)
# 
# `pair_seen` flag handles generalization: if a pair is not in TwoSIDES, the model uses drug-level features only.
# 



# Code Cell: code





# Code Cell: code

# Cell 3.1 | Filter OffSIDES — remove weak / unreliable signals
# A < 5        : fewer than 5 reports support this signal — statistically too noisy
# PRR_error NA : PRR estimate is unreliable (PSM matching failed)
# PRR <= 0     : invalid signal

offs_clean = offsides[
    (offsides['A'] >= 2) &
    (offsides['PRR_error'].notna()) &
    (offsides['PRR'] > 0)
].copy()

retained = 100 * len(offs_clean) / len(offsides)
print(f"Before: {len(offsides):,}  |  After: {len(offs_clean):,}  ({retained:.1f}% retained)")



# Code Cell: code

# Cell 3.2 | Build drug-level feature table from OffSIDES
# For each drug, summarize its adverse event profile across ALL its side effects.
# These features capture intrinsic drug toxicity — how dangerous a drug is alone.
# A drug with high avg_prr and many side effects is a known problematic drug.

drug_features = offs_clean.groupby('drug_rxnorn_id').agg(
    drug_avg_prr          = ('PRR',                     'mean'),
    drug_max_prr          = ('PRR',                     'max'),
    drug_prr_std          = ('PRR',                     'std'),
    drug_num_side_effects = ('condition_meddra_id',     'nunique'),
    drug_mean_rep_freq    = ('mean_reporting_frequency', 'mean'),
    drug_total_A          = ('A',                       'sum'),
    drug_avg_prr_error    = ('PRR_error',               'mean'),
).reset_index()

drug_features['drug_prr_std'] = drug_features['drug_prr_std'].fillna(0)

# Drug name lookup: rxnorm_id -> concept_name (needed for RRS demo)
drug_name_lookup = (
    offs_clean[['drug_rxnorn_id', 'drug_concept_name']]
    .drop_duplicates()
    .set_index('drug_rxnorn_id')['drug_concept_name']
    .to_dict()
)

print(f"Drug-level feature table : {drug_features.shape}")
print(f"Drugs in name lookup     : {len(drug_name_lookup):,}")
drug_features.head(3)



# Code Cell: code

# Cell 3.3 | Build pair-level feature table from TwoSIDES (full dataset, chunked)

# Step 1: detect actual column names once before the loop
header_two = pd.read_csv('data/TWOSIDES.csv.gz', compression='gzip', nrows=0)
print("Actual TwoSIDES columns:", list(header_two.columns))

COL_MAP_TWO = {}
for col in header_two.columns:
    c = col.strip().lower()
    if   'rxnorn' in c and '1' in c:                          COL_MAP_TWO[col] = 'drug_1_rxnorm_id'
    elif 'rxnorm' in c and '2' in c:                          COL_MAP_TWO[col] = 'drug_2_rxnorm_id'
    elif 'concept_name' in c and '1' in c:                    COL_MAP_TWO[col] = 'drug_1_concept_name'
    elif 'concept_name' in c and ('2' in c or '3' in c):      COL_MAP_TWO[col] = 'drug_2_concept_name'
    elif 'meddra' in c:                                       COL_MAP_TWO[col] = 'condition_meddra_id'
    elif 'concept_name' in c:                                 COL_MAP_TWO[col] = 'condition_concept_name'
    elif c == 'a':                                            COL_MAP_TWO[col] = 'A'
    elif c == 'b':                                            COL_MAP_TWO[col] = 'B'
    elif c == 'c':                                            COL_MAP_TWO[col] = 'C'
    elif c == 'd':                                            COL_MAP_TWO[col] = 'D'
    elif c == 'prr' and 'error' not in c:                     COL_MAP_TWO[col] = 'PRR'
    elif 'prr_error' in c or 'prr error' in c:                COL_MAP_TWO[col] = 'PRR_error'
    elif 'reporting_frequency' in c or 'reporting freq' in c: COL_MAP_TWO[col] = 'mean_reporting_frequency'

print("\nColumn map:")
for k, v in COL_MAP_TWO.items():
    print(f"  {repr(k):<45} -> {repr(v)}")

NUM_COLS = ['A', 'B', 'C', 'D', 'PRR', 'PRR_error', 'mean_reporting_frequency']

# Step 2: chunked processing using actual column names
pair_chunks = []

for i, chunk in enumerate(pd.read_csv(
        'data/TWOSIDES.csv.gz',
        chunksize=300_000,
        usecols=list(COL_MAP_TWO.keys()),
        compression='gzip')):

    # Rename to standard names
    chunk = chunk.rename(columns=COL_MAP_TWO)

    # Coerce numeric columns — must happen before any comparison
    for col in NUM_COLS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

    # Now safe to filter
    chunk = chunk[
        (chunk['A'] >= 5) &
        (chunk['PRR_error'].notna()) &
        (chunk['PRR'] > 0)
    ]
    chunk = chunk[chunk['condition_concept_name'].str.lower() != 'unevaluable event']

    if len(chunk) == 0:
        continue

    # Top side effect per pair = row with highest PRR in this chunk
    idx_max = chunk.groupby(['drug_1_rxnorm_id', 'drug_2_rxnorm_id'])['PRR'].idxmax()
    top_fx  = chunk.loc[idx_max, ['drug_1_rxnorm_id', 'drug_2_rxnorm_id',
                                   'condition_concept_name']].copy()
    top_fx  = top_fx.rename(columns={'condition_concept_name': 'top_effect'})

    agg = chunk.groupby(['drug_1_rxnorm_id', 'drug_2_rxnorm_id']).agg(
        pair_avg_prr       = ('PRR',                      'mean'),
        pair_max_prr       = ('PRR',                      'max'),
        pair_prr_std       = ('PRR',                      'std'),
        pair_num_effects   = ('condition_meddra_id',      'nunique'),
        pair_A_sum         = ('A',                        'sum'),
        pair_B_sum         = ('B',                        'sum'),
        pair_C_sum         = ('C',                        'sum'),
        pair_D_sum         = ('D',                        'sum'),
        pair_avg_prr_error = ('PRR_error',                'mean'),
        pair_avg_rep_freq  = ('mean_reporting_frequency', 'mean'),
    ).reset_index()

    agg = agg.merge(top_fx, on=['drug_1_rxnorm_id', 'drug_2_rxnorm_id'], how='left')
    pair_chunks.append(agg)

    if (i + 1) % 10 == 0:
        print(f"  Processed chunk {i+1}...")

print("Concatenating and re-aggregating across chunks...")
raw = pd.concat(pair_chunks, ignore_index=True)
del pair_chunks; gc.collect()

pair_features = raw.groupby(['drug_1_rxnorm_id', 'drug_2_rxnorm_id']).agg({
    'pair_avg_prr'      : 'mean',
    'pair_max_prr'      : 'max',
    'pair_prr_std'      : 'mean',
    'pair_num_effects'  : 'sum',
    'pair_A_sum'        : 'sum',
    'pair_B_sum'        : 'sum',
    'pair_C_sum'        : 'sum',
    'pair_D_sum'        : 'sum',
    'pair_avg_prr_error': 'mean',
    'pair_avg_rep_freq' : 'mean',
    'top_effect'        : 'first',
}).reset_index()

pair_features['pair_prr_std'] = pair_features['pair_prr_std'].fillna(0)
del raw; gc.collect()

print(f"\nPair-level feature table : {pair_features.shape}")
print(f"Unique drug pairs        : {len(pair_features):,}")
pair_features.head(3)




# Code Cell: code

# Standardize drug_features column name first
drug_features = drug_features.rename(
    columns={'drug_rxnorn_id': 'drug_rxnorm_id'}
)



# Code Cell: code

# Cell 3.4 | Join pair features with drug-level features
# For every pair (drug_1, drug_2), attach both drugs' individual profiles.
# Prefixed drug1_* and drug2_* to avoid column name collision.

df = pair_features.merge(
    drug_features.add_prefix('drug1_').rename(
        columns={'drug1_drug_rxnorm_id': 'drug_1_rxnorm_id'}),
    on='drug_1_rxnorm_id', how='left'
).merge(
    drug_features.add_prefix('drug2_').rename(
        columns={'drug2_drug_rxnorm_id': 'drug_2_rxnorm_id'}),
    on='drug_2_rxnorm_id', how='left'
)

print(f"Merged shape : {df.shape}")
print(f"Columns      : {list(df.columns)}")



# Code Cell: code

# DEBUG | Run before Cell 3.5 to diagnose the KeyError

print("=== df columns ===")
for col in sorted(df.columns):
    print(f"  {col}")

print(f"\nTotal columns: {len(df.columns)}")
print(f"\ndf shape: {df.shape}")

# Specifically check if the join worked
drug1_cols = [c for c in df.columns if 'drug1' in c]
drug2_cols = [c for c in df.columns if 'drug2' in c]
pair_cols  = [c for c in df.columns if 'pair' in c]

print(f"\ndrug1_* columns ({len(drug1_cols)}): {drug1_cols}")
print(f"drug2_* columns ({len(drug2_cols)}): {drug2_cols}")
print(f"pair_*  columns ({len(pair_cols)}):  {pair_cols}")



# Code Cell: code

# Cell 3.5 | Derived statistical features
# These features capture relationships not directly present in raw columns.
# They are the heart of the feature engineering contribution.

EPS = 1e-6

# log(PRR): right-skewed distribution — log transform normalizes for linear models
df['log_pair_avg_prr'] = np.log(df['pair_avg_prr'] + EPS)
df['log_pair_max_prr'] = np.log(df['pair_max_prr'] + EPS)

# Signal-to-noise ratio: high PRR with low error = reliable signal
# Low PRR_error means the PSM matching produced a stable estimate
df['pair_snr']  = df['pair_avg_prr']  / (df['pair_avg_prr_error']  + EPS)
df['drug1_snr'] = df['drug1_drug_avg_prr'] / (df['drug1_drug_avg_prr_error'] + EPS)
df['drug2_snr'] = df['drug2_drug_avg_prr'] / (df['drug2_drug_avg_prr_error'] + EPS)

# NOVEL — Relative Risk Ratio:
# Does this PAIR's risk exceed what you'd predict from each drug individually?
# RRR > 1 means the combination is synergistically more dangerous.
# RRR < 1 means the pair is actually less risky than expected (antagonism).
df['relative_risk_ratio'] = (
    df['pair_avg_prr'] /
    (df['drug1_drug_avg_prr'] * df['drug2_drug_avg_prr'] + EPS)
)

# Report confidence: fraction of all relevant FAERS reports supporting this signal
df['report_confidence'] = (
    df['pair_A_sum'] /
    (df['pair_A_sum'] + df['pair_B_sum'] + df['pair_C_sum'] + df['pair_D_sum'] + EPS)
)

# Effect diversity ratio: how many types of side effects does this pair trigger?
# Pairs that cause many different types of effects are broadly dangerous.
df['effect_diversity_ratio'] = (
    df['pair_num_effects'] /
    (df['drug1_drug_num_side_effects'] + df['drug2_drug_num_side_effects'] + EPS)
)

print("Derived features summary:")
print(df[['pair_snr','relative_risk_ratio',
          'report_confidence','effect_diversity_ratio']].describe().round(4))



# Code Cell: code

pair_features.columns



# Code Cell: code

drug_features.columns



# Code Cell: code

df.columns



# Code Cell: code

FEATURE_COLS = [
    # Pair stats (excluding direct PRR)
    'pair_prr_std',
    'pair_num_effects',
    'pair_A_sum',
    'pair_B_sum',
    'pair_C_sum',
    'pair_D_sum',
    'pair_avg_prr_error',
    'pair_avg_rep_freq',

    # Drug-level features
    'drug1_drug_avg_prr',
    'drug2_drug_avg_prr',
    'drug1_drug_num_side_effects',
    'drug2_drug_num_side_effects',
    'drug1_snr',
    'drug2_snr',

    # Engineered features
    'pair_snr',
    'relative_risk_ratio',
    'report_confidence',
    'effect_diversity_ratio'
]
X = df[FEATURE_COLS]
y = np.log1p(df['pair_avg_prr'])



# Code Cell: code

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)



# Code Cell: code

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)



# Code Cell: code

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE :", mae)
print("R²  :", r2)



# markdown Cell

# *PREDICTION*



# Code Cell: code

# Drug-level features extracted from df
drug1_cols = [c for c in df.columns if c.startswith("drug1_")]
drug2_cols = [c for c in df.columns if c.startswith("drug2_")]

# Create unified drug feature table
drug_features_lookup = {}

for _, row in df.iterrows():
    d1 = row['drug_1_rxnorm_id']
    d2 = row['drug_2_rxnorm_id']

    if d1 not in drug_features_lookup:
        drug_features_lookup[d1] = {
            k.replace("drug1_", ""): row[k] for k in drug1_cols
        }

    if d2 not in drug_features_lookup:
        drug_features_lookup[d2] = {
            k.replace("drug2_", ""): row[k] for k in drug2_cols
        }



# Code Cell: code

pair_lookup = df.set_index(
    ['drug_1_rxnorm_id', 'drug_2_rxnorm_id']
)



# Code Cell: code

import numpy as np

def predict_ddi_regression(drug1_id, drug2_id):

    # Ensure consistent ordering (important)
    key = (drug1_id, drug2_id)
    rev_key = (drug2_id, drug1_id)

    if key in pair_lookup.index:
        row = pair_lookup.loc[key]
    elif rev_key in pair_lookup.index:
        row = pair_lookup.loc[rev_key]
    else:
        print("Pair not seen before — estimating from individual drugs")

        if drug1_id not in drug_features_lookup or drug2_id not in drug_features_lookup:
            print("One or both drugs not found.")
            return None

        # Construct synthetic pair feature
        row = {}
        d1 = drug_features_lookup[drug1_id]
        d2 = drug_features_lookup[drug2_id]

        # Minimal fallback logic
        row['pair_prr_std'] = 0
        row['pair_num_effects'] = 1
        row['pair_A_sum'] = 1
        row['pair_B_sum'] = 1
        row['pair_C_sum'] = 1
        row['pair_D_sum'] = 1
        row['pair_avg_prr_error'] = 1
        row['pair_avg_rep_freq'] = 0.01

        row['drug1_drug_avg_prr'] = d1['drug_avg_prr']
        row['drug2_drug_avg_prr'] = d2['drug_avg_prr']
        row['drug1_drug_num_side_effects'] = d1['drug_num_side_effects']
        row['drug2_drug_num_side_effects'] = d2['drug_num_side_effects']
        row['drug1_snr'] = d1['snr']
        row['drug2_snr'] = d2['snr']

        row['pair_snr'] = (d1['drug_avg_prr'] + d2['drug_avg_prr']) / 2
        row['relative_risk_ratio'] = 1
        row['report_confidence'] = 0.01
        row['effect_diversity_ratio'] = 0.001

    # Convert to dataframe
    if isinstance(row, dict):
        X_input = np.array([[row[col] for col in FEATURE_COLS]])
    else:
        X_input = np.array([[row[col] for col in FEATURE_COLS]])

    # Predict log(PRR)
    log_prr_pred = rf.predict(X_input)[0]

    # Convert back
    prr_pred = np.expm1(log_prr_pred)

    print("Predicted log(PRR):", round(log_prr_pred, 4))
    print("Predicted PRR     :", round(prr_pred, 4))

    return prr_pred



# Code Cell: code

predict_ddi_regression(99, 519)



# Code Cell: code

pair_features.head()



# Code Cell: code

import joblib

# Save model
joblib.dump(rf, "ddi_random_forest_regressor.pkl")

print("Model saved successfully.")



# Code Cell: code

from google.colab import files

files.download("ddi_random_forest_regressor.pkl")



# markdown Cell

# **FOR LATER**
# 
# 



# Code Cell: code

# Cell 3.6 | Define label + final feature matrix
# Y = 1 if MAX PRR for this pair >= PRR_THRESHOLD (2.0)
# Meaning: at least one side effect showed a significantly elevated reporting ratio
# for this drug combination vs. PSM-matched controls.

df['label'] = (df['pair_max_prr'] >= PRR_THRESHOLD).astype(int)

FEATURE_COLS = [
    # Pair-level raw features
    'pair_avg_prr', 'pair_max_prr', 'pair_prr_std',
    'pair_num_effects', 'pair_A_sum', 'pair_B_sum',
    'pair_avg_prr_error', 'pair_avg_rep_freq',
    # Drug 1 individual features
    'drug1_drug_avg_prr', 'drug1_drug_max_prr', 'drug1_drug_prr_std',
    'drug1_drug_num_side_effects', 'drug1_drug_mean_rep_freq', 'drug1_drug_total_A',
    # Drug 2 individual features
    'drug2_drug_avg_prr', 'drug2_drug_max_prr', 'drug2_drug_prr_std',
    'drug2_drug_num_side_effects', 'drug2_drug_mean_rep_freq', 'drug2_drug_total_A',
    # Derived statistical features
    'log_pair_avg_prr', 'log_pair_max_prr',
    'pair_snr', 'drug1_snr', 'drug2_snr',
    'relative_risk_ratio', 'report_confidence', 'effect_diversity_ratio',
]

X_raw = df[FEATURE_COLS].copy()
y     = df['label'].copy()

print(f"Feature matrix : {X_raw.shape}")
print(f"Positive pairs : {y.sum():,}  ({100*y.mean():.2f}%)")
print(f"Negative pairs : {(1-y).sum():,}  ({100*(1-y.mean()):.2f}%)")



# Code Cell: code

# Cell 3.7 | Feature correlation heatmap
# Identifies multicollinearity — important context for interpreting
# feature importances and understanding why certain features dominate.

plt.figure(figsize=(14, 11))
corr = X_raw.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
            annot=False, linewidths=0.3, vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix', fontsize=13)
plt.tight_layout()
plt.savefig('feature_correlation.png', bbox_inches='tight')
plt.show()



# Code Cell: code

# Cell 3.8 | Train / test split + imputation
# Stratified split: preserves the class ratio in both sets.
# Critical when class imbalance is severe — random split could
# accidentally under-represent the positive class in the test set.
#
# Imputer: median imputation for pairs/drugs missing from OFFSIDES.
# IMPORTANT: fit imputer only on training data to avoid data leakage.

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=42
)

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(
    imputer.fit_transform(X_train_raw), columns=FEATURE_COLS)
X_test  = pd.DataFrame(
    imputer.transform(X_test_raw),      columns=FEATURE_COLS)

print(f"Train : {X_train.shape}  positive={y_train.mean():.3f}")
print(f"Test  : {X_test.shape}   positive={y_test.mean():.3f}")
print("Imputer fit on training data only — no leakage into test set.")



# markdown Cell

# ---
# ## 🏋️ Phase 4 — Baseline: Logistic Regression
# 
# **Why use it here:**  
# Interpretable coefficients directly show each feature's directional contribution to interaction risk. Fast to train. Establishes the performance floor for comparison.
# 
# **Why it won't be our best model:**  
# Assumes a linear decision boundary. PRR-based interactions are fundamentally nonlinear — e.g., the combination of high `pair_max_prr` AND low `pair_avg_prr_error` produces a synergistic risk signal that LR can't capture without explicit manual feature crossing. Literature shows LR underperforms RF/XGBoost by 12–22% on DDI tabular benchmarks.
# 



# Code Cell: code

# Cell 4.1 | Train Logistic Regression
# Pipeline ensures StandardScaler sees only training data (no leakage).
# class_weight='balanced': automatically up-weights the minority class
# proportional to its frequency: weight = n_samples / (n_classes * class_count)

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        C=1.0  # moderate L2 regularization
    ))
])

lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
lr_pred  = lr_pipe.predict(X_test)

lr_roc = roc_auc_score(y_test, lr_proba)
lr_prc = average_precision_score(y_test, lr_proba)
lr_f1  = f1_score(y_test, lr_pred)

print(f"ROC-AUC : {lr_roc:.4f}")
print(f"PR-AUC  : {lr_prc:.4f}")
print(f"F1      : {lr_f1:.4f}")
print()
print(classification_report(y_test, lr_pred, target_names=['No Signal', 'Signal']))



# Code Cell: code

# Cell 4.2 | Logistic Regression — feature coefficients
# Positive coefficient = feature increases risk probability
# This is LR's key advantage: direct, linear interpretability

coefs = pd.DataFrame({
    'Feature'    : FEATURE_COLS,
    'Coefficient': lr_pipe.named_steps['clf'].coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False).head(20)

colors = ['#E05C4C' if c > 0 else '#4C8BE0' for c in coefs['Coefficient']]

plt.figure(figsize=(10, 7))
plt.barh(coefs['Feature'], coefs['Coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Logistic Regression — Top 20 Feature Coefficients', fontsize=12)
plt.xlabel('Coefficient (positive = increases predicted risk)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('lr_coefficients.png', bbox_inches='tight')
plt.show()



# markdown Cell

# ---
# ## 🌲 Phase 5 — Random Forest
# 
# **Why use it here:**  
# Tree-based models require no feature scaling and naturally capture nonlinear thresholds (e.g., PRR crosses from 'noise' to 'signal' at a specific value — a tree learns this directly as a split). Bagging across 300 trees reduces variance significantly, which matters for noisy FAERS-derived data. MDI feature importance is free at inference time.
# 
# **SMOTE here (not `scale_pos_weight`):**  
# For RF, SMOTE is preferred because it creates synthetic minority-class samples that the individual trees can learn from during training. `scale_pos_weight` is a XGBoost-specific loss function modifier — using it with RF does not have the same effect.
# 



# Code Cell: code

# Cell 5.1 | Train Random Forest with SMOTE
# SMOTE is applied inside ImbPipeline — fit only on training folds,
# never applied to the test set. This prevents data leakage.
# k_neighbors=5: for each minority sample, creates synthetic samples
# by interpolating with its 5 nearest minority neighbors in feature space.
#
# max_features='sqrt': at each tree split, consider sqrt(n_features) candidates.
# This decorrelates the trees and is standard RF best practice.

rf_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ))
])

print("Training Random Forest with SMOTE (2–5 minutes)...")
rf_pipe.fit(X_train, y_train)
rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
rf_pred  = rf_pipe.predict(X_test)

rf_roc = roc_auc_score(y_test, rf_proba)
rf_prc = average_precision_score(y_test, rf_proba)
rf_f1  = f1_score(y_test, rf_pred)

print(f"ROC-AUC : {rf_roc:.4f}")
print(f"PR-AUC  : {rf_prc:.4f}")
print(f"F1      : {rf_f1:.4f}")
print()
print(classification_report(y_test, rf_pred, target_names=['No Signal', 'Signal']))



# Code Cell: code

# Cell 5.2 | Random Forest — MDI feature importance
# Mean Decrease Impurity: average reduction in node impurity (Gini)
# weighted by the number of samples reaching that node, across all trees.
# Higher = more useful for separating signal from no-signal at each split.

rf_clf = rf_pipe.named_steps['clf']
feat_imp = pd.DataFrame({
    'Feature'   : FEATURE_COLS,
    'Importance': rf_clf.feature_importances_
}).sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(10, 7))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title('Random Forest — Top 20 Feature Importances (MDI)', fontsize=12)
plt.xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig('rf_importance.png', bbox_inches='tight')
plt.show()



# markdown Cell

# ---
# ## ⚡ Phase 6 — XGBoost (Best Model)
# 
# **Why use it here:**  
# Gradient boosting trains trees *sequentially* — each new tree targets the residual errors of the previous ensemble. This is fundamentally more powerful than RF's parallel bagging for learning complex patterns.
# 
# **`scale_pos_weight`:** Set to `neg_count / pos_count`. This directly modifies the gradient computation — positive-class misclassifications receive proportionally higher gradient updates. More principled than SMOTE for boosting models.
# 
# **`eval_metric='aucpr'`:** Optimizes for PR-AUC during early stopping — the correct metric for imbalanced classes, unlike log-loss or accuracy.
# 
# **`early_stopping_rounds=30`:** Stops training when PR-AUC on the eval set hasn't improved for 30 rounds — prevents overfitting automatically.
# 



# Code Cell: code

# Cell 6.1 | Train XGBoost (default params)
neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
spw = neg_count / pos_count
print(f"scale_pos_weight = {spw:.2f}  (neg:pos ratio in training set)")

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,          # row sampling per tree — reduces overfitting
    colsample_bytree=0.8,   # feature sampling per tree — decorrelates trees
    scale_pos_weight=spw,
    reg_alpha=0.1,          # L1 regularization — sparse feature weights
    reg_lambda=1.0,         # L2 regularization — smooth feature weights
    eval_metric='aucpr',
    early_stopping_rounds=30,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

print("Training XGBoost...")
xgb_clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
xgb_pred  = xgb_clf.predict(X_test)

xgb_roc = roc_auc_score(y_test, xgb_proba)
xgb_prc = average_precision_score(y_test, xgb_proba)
xgb_f1  = f1_score(y_test, xgb_pred)

print(f"ROC-AUC        : {xgb_roc:.4f}")
print(f"PR-AUC         : {xgb_prc:.4f}")
print(f"F1             : {xgb_f1:.4f}")
print(f"Best iteration : {xgb_clf.best_iteration}")
print()
print(classification_report(y_test, xgb_pred, target_names=['No Signal', 'Signal']))



# Code Cell: code

# Cell 6.2 | XGBoost — hyperparameter tuning with RandomizedSearchCV
# RandomizedSearchCV randomly samples n_iter combinations from param_dist.
# Faster than GridSearchCV for large parameter spaces.
# 3-fold stratified CV on training data — test set is NEVER touched during tuning.

param_dist = {
    'max_depth'        : [4, 6, 8, 10],
    'learning_rate'    : [0.01, 0.05, 0.1],
    'n_estimators'     : [200, 400, 600],
    'subsample'        : [0.7, 0.8, 0.9],
    'colsample_bytree' : [0.7, 0.8, 1.0],
    'reg_alpha'        : [0, 0.1, 0.5],
    'reg_lambda'       : [0.5, 1.0, 2.0],
}

base_xgb = xgb.XGBClassifier(
    scale_pos_weight=spw,
    eval_metric='aucpr',
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

rscv = RandomizedSearchCV(
    base_xgb,
    param_distributions=param_dist,
    n_iter=25,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Running RandomizedSearchCV (25 iterations x 3-fold CV)...")
rscv.fit(X_train, y_train)

print(f"\nBest CV ROC-AUC : {rscv.best_score_:.4f}")
print(f"Best params     : {rscv.best_params_}")

best_xgb   = rscv.best_estimator_
best_proba = best_xgb.predict_proba(X_test)[:, 1]
best_pred  = best_xgb.predict(X_test)

best_roc = roc_auc_score(y_test, best_proba)
best_prc = average_precision_score(y_test, best_proba)
best_f1  = f1_score(y_test, best_pred)

print(f"\nTuned XGBoost — Test ROC-AUC : {best_roc:.4f}")
print(f"Tuned XGBoost — Test PR-AUC  : {best_prc:.4f}")
print(f"Tuned XGBoost — Test F1      : {best_f1:.4f}")



# markdown Cell

# ---
# ## 📊 Phase 7 — Model Comparison



# Code Cell: code

# Cell 7.1 | ROC and Precision-Recall curves — all models
# ROC-AUC : area under TPR vs FPR curve — measures overall ranking ability
# PR-AUC  : area under Precision vs Recall curve — PREFERRED for imbalanced data
#           A model can look good on ROC but fail on PR when positives are rare.
#           Always report both.

models = {
    'Logistic Regression': (lr_proba,   lr_roc,  lr_prc),
    'Random Forest'      : (rf_proba,   rf_roc,  rf_prc),
    'XGBoost (default)'  : (xgb_proba,  xgb_roc, xgb_prc),
    'XGBoost (tuned)'    : (best_proba, best_roc, best_prc),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for name, (proba, roc, prc) in models.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc:.3f})")

    p, r, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(r, p, linewidth=2, label=f"{name} (AP={prc:.3f})")

axes[0].plot([0,1],[0,1], 'k--', linewidth=0.8, label='Random')
axes[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curves')
axes[0].legend(fontsize=8)

axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curves')
axes[1].legend(fontsize=8)

plt.suptitle('Model Comparison — DDI Signal Classification', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('model_comparison.png', bbox_inches='tight')
plt.show()



# Code Cell: code

# Cell 7.2 | Metrics summary table
rows = []
for name, (proba, roc, prc) in models.items():
    pred = (proba >= 0.5).astype(int)
    rows.append({
        'Model'  : name,
        'ROC-AUC': round(roc, 4),
        'PR-AUC' : round(prc, 4),
        'F1'     : round(f1_score(y_test, pred), 4),
    })

results_df = pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False)
print(results_df.to_string(index=False))



# Code Cell: code

# Cell 7.3 | Confusion matrix — best model
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: No Signal', 'Pred: Signal'],
            yticklabels=['True: No Signal', 'True: Signal'])
plt.title('Confusion Matrix — Best XGBoost (Tuned)', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()



# markdown Cell

# ---
# ## 🔎 Phase 8 — SHAP Explainability
# 
# **Why SHAP:**  
# SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory. Each feature's SHAP value is its average marginal contribution across all possible orderings of features — the theoretically correct way to attribute prediction credit.
# 
# **Clinical relevance:**  
# A 2025 systematic review of 147 DDI studies specifically identifies SHAP and LIME as critical techniques for bridging the gap between model performance and clinical trustworthiness. Clinicians need to know *why* a drug pair was flagged, not just *that* it was flagged. SHAP powers the explanation layer in the RRS demo.
# 
# **TreeExplainer:**  
# The fast, exact SHAP algorithm for tree-based models (RF and XGBoost). Computes exact Shapley values by exploiting the tree structure — no approximation needed.
# 



# Code Cell: code

# Cell 8.1 | Compute SHAP values for best XGBoost
# Using a 2000-sample subset for Colab speed.
# TreeExplainer is the exact SHAP method for tree-based models.

shap.initjs()

explainer  = shap.TreeExplainer(best_xgb)
n_shap     = min(2000, len(X_test))
X_shap     = X_test.sample(n_shap, random_state=42)
shap_vals  = explainer.shap_values(X_shap)

print(f"SHAP values computed for {n_shap} test samples")
print(f"Shape : {shap_vals.shape}  (samples x features)")



# Code Cell: code

# Cell 8.2 | SHAP summary plot — global feature importance
# Each dot = one sample.
# X-axis position = SHAP value (positive = pushed toward 'Signal')
# Color = feature value (red=high, blue=low)
# Features ranked by mean |SHAP value| — the true global importance.

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_shap, feature_names=FEATURE_COLS,
                  show=False, plot_size=(10, 8))
plt.title('SHAP Summary — Feature Impact on DDI Risk Prediction', fontsize=12)
plt.tight_layout()
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.show()



# Code Cell: code

# Cell 8.3 | SHAP force plot — single highest-risk prediction
# Shows the exact contribution of every feature for one specific pair.
# Prototype for the per-pair explanation in the RRS demo.

top_idx   = int(np.argmax(best_proba))
top_X     = X_test.iloc[[top_idx]]
top_shap  = explainer.shap_values(top_X)

print(f"Predicted probability : {best_proba[top_idx]:.4f}")
print(f"True label            : {y_test.iloc[top_idx]}")
print()

local_df = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'SHAP'   : top_shap[0],
    'Value'  : top_X.values[0]
}).sort_values('SHAP', key=abs, ascending=False).head(10)

print("Top 10 features driving this prediction:")
print(local_df.to_string(index=False))

shap.force_plot(
    explainer.expected_value,
    top_shap[0],
    top_X,
    feature_names=FEATURE_COLS,
    matplotlib=True
)
plt.title('SHAP Force Plot — Highest-Risk Pair')
plt.savefig('shap_force_plot.png', bbox_inches='tight')
plt.show()



# markdown Cell

# ---
# ## 🧮 Phase 9 — Regimen Risk Score (RRS) Engine
# 
# ### The Novel Contribution
# 
# All existing work (Decagon, PolyLLM, DrugBank classifiers) predicts **one pair at a time**. Real patients take 5–10 drugs simultaneously. The RRS is an original formulation that aggregates pairwise risk across a full medication list.
# 
# ### Formula
# 
# ```
# RRS = ( SUM_ij [ P_ij * w_ij ] / C(n,2) ) * 10
# 
#   P_ij  = model-predicted risk probability for pair (drug_i, drug_j)
#   w_ij  = confidence weight = SNR / (SNR + 5)   [sigmoid-like: clips to 0-1]
#            * 0.5 if pair is unseen in TwoSIDES   [penalize lower certainty]
#   C(n,2) = n*(n-1)/2 = total possible pairs (normalization)
#   x10    = scale output to readable 0–10 range
# ```
# 
# ### Troublemaker Score per drug k
# 
# ```
# TS_k = mean( P_kj  for all j != k )
# ```
# 
# The drug with the highest TS is the interaction hub — the one most worth reviewing first.
# 



# Code Cell: code

# Cell 9.1 | Feature extractor for any drug pair
def get_pair_features(rxnorm_1, rxnorm_2):
    """
    Build the 28-feature vector for a given drug pair.

    Returns:
        feat_df   : pd.DataFrame (1 row, FEATURE_COLS columns)
        top_effect: str — name of the most frequently flagged side effect
        pair_seen : int — 1 if pair exists in TwoSIDES, 0 if unseen
    """
    EPS = 1e-6

    pair = pair_features[
        ((pair_features['drug_1_rxnorm_id'] == rxnorm_1) &
         (pair_features['drug_2_rxnorm_id'] == rxnorm_2)) |
        ((pair_features['drug_1_rxnorm_id'] == rxnorm_2) &
         (pair_features['drug_2_rxnorm_id'] == rxnorm_1))
    ]

    d1 = drug_features[drug_features['drug_rxnorm_id'] == rxnorm_1]
    d2 = drug_features[drug_features['drug_rxnorm_id'] == rxnorm_2]

    row = {}
    pair_c = pair_d = 0

    if len(pair) > 0:
        p = pair.iloc[0]
        row.update({
            'pair_avg_prr'      : p['pair_avg_prr'],
            'pair_max_prr'      : p['pair_max_prr'],
            'pair_prr_std'      : p['pair_prr_std'],
            'pair_num_effects'  : p['pair_num_effects'],
            'pair_A_sum'        : p['pair_A_sum'],
            'pair_B_sum'        : p['pair_B_sum'],
            'pair_avg_prr_error': p['pair_avg_prr_error'],
            'pair_avg_rep_freq' : p['pair_avg_rep_freq'],
        })
        pair_c   = p.get('pair_C_sum', 0)
        pair_d   = p.get('pair_D_sum', 0)
        top_fx   = p.get('top_effect', 'Unknown')
        pair_seen = 1
    else:
        # Unseen pair: zero out pair-level features, flag as unseen
        row.update({
            'pair_avg_prr': 0, 'pair_max_prr': 0, 'pair_prr_std': 0,
            'pair_num_effects': 0, 'pair_A_sum': 0, 'pair_B_sum': 0,
            'pair_avg_prr_error': 1, 'pair_avg_rep_freq': 0,
        })
        top_fx = 'Unknown'; pair_seen = 0

    # Attach individual drug profiles
    for prefix, drug_row in [('drug1', d1), ('drug2', d2)]:
        if len(drug_row) > 0:
            dr = drug_row.iloc[0]
            row[f'{prefix}_avg_prr']          = dr['drug_avg_prr']
            row[f'{prefix}_max_prr']          = dr['drug_max_prr']
            row[f'{prefix}_prr_std']          = dr['drug_prr_std']
            row[f'{prefix}_num_side_effects'] = dr['drug_num_side_effects']
            row[f'{prefix}_mean_rep_freq']    = dr['drug_mean_rep_freq']
            row[f'{prefix}_total_A']          = dr['drug_total_A']
            row[f'{prefix}_avg_prr_error']    = dr['drug_avg_prr_error']
        else:
            for k in ['avg_prr','max_prr','prr_std','num_side_effects',
                      'mean_rep_freq','total_A']:
                row[f'{prefix}_{k}'] = 0
            row[f'{prefix}_avg_prr_error'] = 1

    # Derived features
    row['log_pair_avg_prr']       = np.log(row['pair_avg_prr'] + EPS)
    row['log_pair_max_prr']       = np.log(row['pair_max_prr'] + EPS)
    row['pair_snr']               = row['pair_avg_prr'] / (row['pair_avg_prr_error'] + EPS)
    row['drug1_snr']              = row['drug1_avg_prr'] / (row['drug1_avg_prr_error'] + EPS)
    row['drug2_snr']              = row['drug2_avg_prr'] / (row['drug2_avg_prr_error'] + EPS)
    row['relative_risk_ratio']    = row['pair_avg_prr'] / (row['drug1_avg_prr'] * row['drug2_avg_prr'] + EPS)
    total_reports                 = row['pair_A_sum'] + row['pair_B_sum'] + pair_c + pair_d + EPS
    row['report_confidence']      = row['pair_A_sum'] / total_reports
    d_eff                         = row['drug1_num_side_effects'] + row['drug2_num_side_effects']
    row['effect_diversity_ratio'] = row['pair_num_effects'] / (d_eff + EPS)

    feat_df = pd.DataFrame([{col: row.get(col, 0) for col in FEATURE_COLS}])
    return feat_df, top_fx, pair_seen


print("get_pair_features() defined")



# Code Cell: code

# Cell 9.2 | RRS scoring engine
def compute_rrs(drug_list, model, imp):
    """
    Compute the Regimen Risk Score (RRS) for a full medication list.

    Args:
        drug_list : list of (rxnorm_id, drug_name) tuples
        model     : trained classifier with predict_proba()
        imp       : fitted SimpleImputer

    Returns:
        dict with rrs, pair_results, troublemaker, ts_scores
    """
    pairs     = list(combinations(drug_list, 2))
    drug_risks = {name: [] for _, name in drug_list}
    pair_results = []

    for (id1, name1), (id2, name2) in pairs:
        feat_df, top_fx, seen = get_pair_features(id1, id2)
        feat_imp = pd.DataFrame(imp.transform(feat_df), columns=FEATURE_COLS)

        prob = float(model.predict_proba(feat_imp)[0, 1])

        # Confidence weight: sigmoid normalization of SNR, halved if pair unseen
        snr = float(feat_imp['pair_snr'].values[0])
        w   = float(np.clip(snr / (snr + 5), 0, 1))
        if seen == 0:
            w *= 0.5

        pair_results.append({
            'drug_1'        : name1,
            'drug_2'        : name2,
            'probability'   : round(prob, 4),
            'confidence'    : round(w, 4),
            'weighted_risk' : round(prob * w, 4),
            'top_effect'    : top_fx,
            'pair_seen'     : seen,
        })
        drug_risks[name1].append(prob)
        drug_risks[name2].append(prob)

    # RRS formula
    total_w  = sum(p['weighted_risk'] for p in pair_results)
    n_pairs  = len(pair_results)
    rrs      = round((total_w / n_pairs) * 10, 2)

    # Troublemaker: drug with highest mean pairwise risk
    ts = {name: round(float(np.mean(risks)), 4)
          for name, risks in drug_risks.items() if risks}
    troublemaker = max(ts, key=ts.get)

    return {
        'rrs'         : rrs,
        'pair_results': sorted(pair_results, key=lambda x: -x['probability']),
        'troublemaker': troublemaker,
        'ts_scores'   : ts,
    }


print("compute_rrs() defined")



# markdown Cell

# ---
# ## 🏥 Phase 10 — RRS End-to-End Demo
# 
# A clinically realistic 6-drug regimen common in elderly patients with cardiovascular disease and type 2 diabetes.



# Code Cell: code

# Cell 10.1 | Drug name -> RxNORM lookup
name_to_rxnorm = {v.lower(): k for k, v in drug_name_lookup.items()}

def find_rxnorm(name):
    """Case-insensitive drug name -> RxNORM ID lookup with fuzzy fallback."""
    result = name_to_rxnorm.get(name.lower())
    if result is None:
        matches = [k for k in name_to_rxnorm if name.lower() in k]
        if matches:
            result = name_to_rxnorm[matches[0]]
            print(f"  Fuzzy: '{name}' -> '{matches[0]}' (ID: {result})")
    return result

# Classic polypharmacy regimen — warfarin interactions are well-documented
REGIMEN_NAMES = ['warfarin', 'aspirin', 'metformin', 'lisinopril',
                 'atorvastatin', 'omeprazole']

regimen = []
for name in REGIMEN_NAMES:
    rxid = find_rxnorm(name)
    if rxid:
        regimen.append((rxid, name.capitalize()))
        print(f"  {name.capitalize():<20} RxNORM: {rxid}")
    else:
        print(f"  WARNING: '{name}' not found in dataset — skipping")

n_pairs = len(list(combinations(regimen, 2)))
print(f"\n{len(regimen)} drugs  |  {n_pairs} pairs to evaluate")



# Code Cell: code

# Cell 10.2 | Run RRS analysis
results = compute_rrs(regimen, best_xgb, imputer)

rrs_val = results['rrs']
level   = ('HIGH RISK' if rrs_val >= 7 else
           'MODERATE RISK' if rrs_val >= 4 else 'LOW RISK')

print("=" * 52)
print(f"  REGIMEN RISK SCORE  :  {rrs_val} / 10")
print(f"  Risk Level          :  {level}")
print(f"  Troublemaker Drug   :  {results['troublemaker']}")
print("=" * 52)
print()

pairs_df = pd.DataFrame(results['pair_results'])
print(pairs_df[['drug_1','drug_2','probability','confidence',
                'top_effect','pair_seen']].to_string(index=False))



# Code Cell: code

# Cell 10.3 | Interaction risk heatmap — portfolio signature visual
# n x n matrix where color intensity = predicted interaction risk.
# Immediately communicates which drug pairs are highest-risk at a glance.
# This is the core visual for the Streamlit demo.

drug_names = [name for _, name in regimen]
n          = len(drug_names)
risk_mat   = np.zeros((n, n))

lookup = {}
for p in results['pair_results']:
    lookup[(p['drug_1'], p['drug_2'])] = p['probability']
    lookup[(p['drug_2'], p['drug_1'])] = p['probability']

for i, d1 in enumerate(drug_names):
    for j, d2 in enumerate(drug_names):
        if i != j:
            risk_mat[i][j] = lookup.get((d1, d2), 0)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(risk_mat, cmap='RdYlGn_r', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Interaction Risk Probability')

ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(drug_names, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(drug_names, fontsize=11)

for i in range(n):
    for j in range(n):
        v = risk_mat[i][j]
        if i != j:
            c = 'white' if v > 0.6 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, color=c, fontweight='bold')

ax.set_title(
    f'Regimen Interaction Risk Heatmap\n'
    f'RRS = {results["rrs"]}/10   |   Troublemaker: {results["troublemaker"]}',
    fontsize=12, pad=15
)
plt.tight_layout()
plt.savefig('rrs_heatmap.png', bbox_inches='tight', dpi=150)
plt.show()



# Code Cell: code

# Cell 10.4 | Troublemaker bar chart
ts_df = pd.DataFrame([
    {'Drug': k, 'Avg Risk': v}
    for k, v in results['ts_scores'].items()
]).sort_values('Avg Risk', ascending=False)

colors = ['#E05C4C' if r['Drug'] == results['troublemaker'] else '#4C8BE0'
          for _, r in ts_df.iterrows()]

plt.figure(figsize=(9, 5))
plt.bar(ts_df['Drug'], ts_df['Avg Risk'], color=colors, edgecolor='white', width=0.6)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Risk = 0.5')
plt.title('Per-Drug Troublemaker Score (Mean Pairwise Interaction Risk)', fontsize=12)
plt.ylabel('Average Pairwise Risk Probability')
plt.xlabel('Drug in Regimen')
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('troublemaker_chart.png', bbox_inches='tight')
plt.show()

print(f"Troublemaker: {results['troublemaker']} — review this drug first.")



# Code Cell: code

# Cell 10.5 | SHAP explanation for the highest-risk pair
top_pair = results['pair_results'][0]
print(f"Explaining: {top_pair['drug_1']} + {top_pair['drug_2']}")
print(f"  Risk probability : {top_pair['probability']}")
print(f"  Confidence       : {top_pair['confidence']}")
print(f"  Top side effect  : {top_pair['top_effect']}")
print(f"  In TwoSIDES      : {'Yes' if top_pair['pair_seen'] else 'No — generalized from drug profiles'}")
print()

id_map = {name: rxid for rxid, name in regimen}
id1 = id_map.get(top_pair['drug_1'])
id2 = id_map.get(top_pair['drug_2'])

if id1 and id2:
    feat_df, _, _ = get_pair_features(id1, id2)
    feat_imp_s    = pd.DataFrame(imputer.transform(feat_df), columns=FEATURE_COLS)
    local_shap    = explainer.shap_values(feat_imp_s)

    local_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'SHAP'   : local_shap[0],
        'Value'  : feat_imp_s.values[0]
    }).sort_values('SHAP', key=abs, ascending=False).head(10)

    print("Why this pair was flagged (top SHAP contributors):")
    print(local_df.to_string(index=False))
else:
    print("Could not map drug names to IDs — check id_map.")



# Code Cell: code

# Cell 10.6 | Save all artifacts for the Streamlit demo
os.makedirs('model_artifacts', exist_ok=True)

with open('model_artifacts/best_xgb.pkl', 'wb')  as f: pickle.dump(best_xgb, f)
with open('model_artifacts/rf_model.pkl', 'wb')   as f: pickle.dump(rf_pipe, f)
with open('model_artifacts/imputer.pkl', 'wb')    as f: pickle.dump(imputer, f)

drug_features.to_parquet('model_artifacts/drug_features.parquet', index=False)
pair_features.to_parquet('model_artifacts/pair_features.parquet', index=False)

with open('model_artifacts/drug_name_lookup.json', 'w') as f:
    json.dump({str(k): v for k, v in drug_name_lookup.items()}, f)

with open('model_artifacts/feature_cols.json', 'w') as f:
    json.dump(FEATURE_COLS, f)

metrics_out = {
    'logistic_regression': {'roc_auc': lr_roc,   'pr_auc': lr_prc,   'f1': lr_f1},
    'random_forest'      : {'roc_auc': rf_roc,   'pr_auc': rf_prc,   'f1': rf_f1},
    'xgboost_default'    : {'roc_auc': xgb_roc,  'pr_auc': xgb_prc,  'f1': xgb_f1},
    'xgboost_tuned'      : {'roc_auc': best_roc, 'pr_auc': best_prc, 'f1': best_f1},
    'prr_threshold'      : PRR_THRESHOLD,
}
with open('model_artifacts/metrics.json', 'w') as f:
    json.dump(metrics_out, f, indent=2)

print("Artifacts saved to model_artifacts/:")
for fname in sorted(os.listdir('model_artifacts')):
    kb = os.path.getsize(f'model_artifacts/{fname}') / 1024
    print(f"  {fname:<45} {kb:>8.1f} KB")



# markdown Cell

# ---
# ## ✅ Summary
# 
# | Phase | Output |
# |-------|--------|
# | 1 — Setup | OFFSIDES + TWOSIDES downloaded from NSIDES S3 |
# | 2 — EDA | PRR distributions, class balance (~1:10 imbalance), top 20 side effects |
# | 3 — Features | 28-feature matrix: pair-level + drug-level + 8 derived statistics |
# | 4 — LR | Baseline model + coefficient chart (performance floor) |
# | 5 — RF | Core ensemble + MDI feature importance plot |
# | 6 — XGBoost | Default + tuned via RandomizedSearchCV (best model) |
# | 7 — Comparison | ROC curves + PR curves + metrics table + confusion matrix |
# | 8 — SHAP | Global summary plot + per-prediction force plot |
# | 9 — RRS Engine | `get_pair_features()` + `compute_rrs()` with confidence weighting |
# | 10 — Demo | Heatmap + troublemaker chart + SHAP explanation + artifact export |
# 
# ---
# 
# ### Next Step: Streamlit Demo
# All artifacts are saved in `model_artifacts/`. The demo app will:
# 1. Accept drug names via autocomplete search
# 2. Build the regimen list interactively
# 3. Compute RRS on submit
# 4. Display the interaction heatmap + troublemaker chart + per-pair SHAP explanation
# 
# ---
# *Dataset: NSIDES (Tatonetti Lab) | Model: XGBoost (tuned) | Novel contribution: Regimen Risk Score (RRS)*
# 
