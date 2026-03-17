"""
Standalone training script — generates all .pkl files needed by the Django app.
Equivalent to running the notebook Cells 1-12 + Cell 17.
No visualization libraries needed (no matplotlib/seaborn).
"""
import re, warnings, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'scholarship_dataset_expanded.csv')
PKL_DIR = os.path.join(SCRIPT_DIR, 'app', 'scholarships', 'pkl')
os.makedirs(PKL_DIR, exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'outputs'), exist_ok=True)

# ══════════════════════════════════════════════════════════════
# Cell 2 — Load Dataset
# ══════════════════════════════════════════════════════════════
print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════
# Cell 3 — Data Cleaning
# ══════════════════════════════════════════════════════════════
YES_VALS = {'yes', 'y', '1', 'true', 'yep'}
NO_VALS  = {'no',  'n', '0', 'false', 'nope', 'nop'}

def clean_binary(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    if s in YES_VALS: return 1.0
    if s in NO_VALS:  return 0.0
    return np.nan

def clean_sex(v):
    if pd.isna(v): return 'Unknown'
    s = str(v).strip().lower()
    return 'Female' if s.startswith('f') else ('Male' if s.startswith('m') else 'Unknown')

def clean_civil(v):
    if pd.isna(v): return 'Unknown'
    return {'single':'Single', 'married':'Married', 'widowed':'Widowed',
            'singe':'Single', 'sngle':'Single'}.get(str(v).strip().lower(), 'Unknown')

def clean_hei(v):
    if pd.isna(v): return 'Unknown'
    s = re.sub(r'[\s._-]+', '', str(v).strip().lower())
    if 'suc' in s or 'stateuniversity' in s: return 'SUC'
    if 'luc' in s or 'localuniversity' in s: return 'LUC'
    if 'private' in s:                       return 'Private HEI'
    return 'Unknown'

def clean_strand(v):
    if pd.isna(v): return 'Unknown'
    s = str(v).strip().lower()
    if 'stem' in s:                              return 'STEM'
    if 'abm' in s and 'tvl' in s:               return 'TVL/ABM'
    if 'abm' in s:                               return 'ABM'
    if 'humss' in s or 'gas' in s or 'academic' in s: return 'HUMSS/GAS'
    if 'tvl' in s:                               return 'TVL'
    return 'Other'

def clean_bar(v):
    if pd.isna(v): return 'Unknown'
    s = str(v).strip().lower()
    return 'Urban' if 'urban' in s else ('Rural' if 'rural' in s else 'Unknown')

def clean_cat(v):
    if pd.isna(v): return 'Unknown'
    s = str(v).strip().lower()
    MAP = {
        'coconut-related':  'Coconut-Related',  'sugarcane-related': 'Sugarcane-Related',
        'engineering':      'Engineering',       'enginnering':       'Engineering',
        'agriculture':      'Agriculture',       'agriulture':        'Agriculture',
        'health sciences':  'Health Sciences',   'health sciencess':  'Health Sciences',
        'health science':   'Health Sciences',   'helath sciences':   'Health Sciences',
        'science':          'Science',           'sience':            'Science',
        'sciience':         'Science',
        'social sciences':  'Social Sciences',   'hospitality':       'Hospitality',
        'it/computing':     'IT/Computing',       'i.t/computing':     'IT/Computing',
        'business':         'Business',          'bussiness':         'Business',
        'buisness':         'Business',          'busines':           'Business',
        'education':        'Education',
    }
    for k, val in MAP.items():
        if k in s: return val
    return s.title()

def clean_region(v):
    if pd.isna(v): return 'Unknown'
    return re.sub(r'\s+', ' ', str(v).strip())

df['sex']               = df['sex'].apply(clean_sex)
df['civil_status']      = df['civil_status'].apply(clean_civil)
df['enrolled_hei_type'] = df['enrolled_hei_type'].apply(clean_hei)
df['shs_strand']        = df['shs_strand'].apply(clean_strand)
df['barangay_type']     = df['barangay_type'].apply(clean_bar)
df['course_category']   = df['course_category'].apply(clean_cat)
df['region']            = df['region'].apply(clean_region)

BINARY_COLS = ['is_solo_parent_dependent', 'is_pwd', 'is_indigenous_people',
               'is_4ps_beneficiary', 'is_ofw_dependent', 'has_existing_scholarship']
for col in BINARY_COLS:
    df[col] = df[col].apply(clean_binary)

NUMERIC_COLS = ['gwa_percentage', 'gwa_numeric_1to5',
                'family_annual_income_php', 'family_size', 'age', 'year_level']
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

mask = df['gwa_percentage'].isna() & df['gwa_numeric_1to5'].notna()
df.loc[mask, 'gwa_percentage'] = 100 - (df.loc[mask, 'gwa_numeric_1to5'] - 1) * 7.5
mask = df['gwa_numeric_1to5'].isna() & df['gwa_percentage'].notna()
df.loc[mask, 'gwa_numeric_1to5'] = (100 - df.loc[mask, 'gwa_percentage']) / 7.5 + 1

print("  Cleaning complete.")

# ══════════════════════════════════════════════════════════════
# Cell 4 — Build Text Profiles
# ══════════════════════════════════════════════════════════════
def build_text_profile(row):
    parts = []
    parts.append(str(row.get('course',              '') or '').lower().strip())
    parts.append(str(row.get('course_category',     '') or '').lower().strip())
    parts.append(str(row.get('shs_strand',          '') or '').lower().strip())
    parts.append(str(row.get('region',              '') or '').lower().strip())
    parts.append(str(row.get('enrolled_hei_type',   '') or '').lower().strip())
    parts.append(str(row.get('barangay_type',       '') or '').lower().strip())
    occ = str(row.get('parents_occupation', '') or '').lower().strip()
    if occ and occ != 'nan':
        parts.append(occ)
    if row.get('is_4ps_beneficiary')       == 1.0: parts += ['4ps_beneficiary']  * 3
    if row.get('is_ofw_dependent')         == 1.0: parts += ['ofw_dependent']    * 2
    if row.get('is_indigenous_people')     == 1.0: parts += ['indigenous_people'] * 2
    if row.get('is_pwd')                   == 1.0: parts += ['pwd']              * 2
    if row.get('is_solo_parent_dependent') == 1.0: parts += ['solo_parent']      * 2
    return ' '.join(p for p in parts if p and p != 'nan')

df['text_profile'] = df.apply(build_text_profile, axis=1)
print("  Text profiles created.")

# ══════════════════════════════════════════════════════════════
# Cell 5 — Feature Engineering
# ══════════════════════════════════════════════════════════════
g   = df['gwa_percentage'].fillna(0)
inc = df['family_annual_income_php'].fillna(0)

def gwa_fine_tier(x):
    if pd.isna(x): return 4
    if x >= 97: return 9
    if x >= 95: return 8
    if x >= 93: return 7
    if x >= 91: return 6
    if x >= 88: return 5
    if x >= 85: return 4
    if x >= 82: return 3
    if x >= 78: return 2
    return 1

def income_tier(x):
    if pd.isna(x): return 3
    if x < 100_000: return 1
    if x < 150_000: return 2
    if x < 200_000: return 3
    if x < 250_000: return 4
    if x < 300_000: return 5
    if x < 400_000: return 6
    if x < 500_000: return 7
    return 8

df['gwa_fine'] = df['gwa_percentage'].apply(gwa_fine_tier)
df['inc_tier'] = df['family_annual_income_php'].apply(income_tier)

DOST_FIELDS = {'Engineering','Health Sciences','Science','IT/Computing','Agriculture'}
df['is_coconut']  = (df['course_category'] == 'Coconut-Related').astype(float)
df['is_sugarcane']= (df['course_category'] == 'Sugarcane-Related').astype(float)
df['is_agri']     = df['course_category'].isin(
                        {'Agriculture','Coconut-Related','Sugarcane-Related'}).astype(float)
df['is_dost']     = df['course_category'].isin(DOST_FIELDS).astype(float)
df['is_stem'] = (df['shs_strand'] == 'STEM').astype(float)
df['is_suc']  = (df['enrolled_hei_type'] == 'SUC').astype(float)
df['is_priv'] = (df['enrolled_hei_type'] == 'Private HEI').astype(float)
df['is_luc']  = (df['enrolled_hei_type'] == 'LUC').astype(float)
df['is_r2']   = (df['region'] == 'Region II').astype(float)
df['is_r6']   = (df['region'] == 'Region VI').astype(float)

df['vuln'] = (df['is_4ps_beneficiary'].fillna(0) +
              df['is_solo_parent_dependent'].fillna(0) +
              df['is_pwd'].fillna(0) +
              df['is_indigenous_people'].fillna(0) +
              df['is_ofw_dependent'].fillna(0))

df['sig_univ']       = (g >= 97.0).astype(float)
df['sig_coll_schol'] = ((g >= 95.0) & (g < 97.0)).astype(float)
df['sig_merit_full'] = ((g >= 96.0) & (g < 99.5)).astype(float)
df['sig_merit_half'] = ((g >= 93.0) & (g < 95.0)).astype(float)
df['sig_coscho']     = df['is_coconut']
df['sig_sida']       = (df['is_sugarcane'] * (inc < 200_000)).astype(float)
df['sig_broed']      = df['is_r2']
df['sig_dost']       = (df['is_dost'] * df['is_stem'] * (g >= 85).astype(float))
df['sig_no_schol']   = (inc > 430_000).astype(float)
df['sig_tes']        = ((inc < 290_000) & df['family_annual_income_php'].notna()
                       ).astype(float) * df['vuln'].clip(0, 1)
df['sig_sikap']      = (inc > 300_000).astype(float) * (1 - df['sig_no_schol'])
df['sig_acef']       = (df['is_agri'] * ((g >= 82) & (g < 95)).astype(float)).astype(float)

df['gwa_x_inc']     = df['gwa_fine'] * df['inc_tier']
df['gwa_x_vuln']    = df['gwa_fine'] * df['vuln']
df['gwa_x_coconut'] = g * df['is_coconut']
df['gwa_x_r2']      = g * df['is_r2']
df['gwa_x_dost']    = g * df['is_dost']
df['gwa_inc_ratio'] = df['gwa_fine'] / (df['inc_tier'] + 0.01)
df['log_income'] = np.log1p(df['family_annual_income_php'].fillna(0))
df['gwa_sq']     = g ** 2
df['inc_sq']     = df['log_income'] ** 2

print(f"  Feature engineering complete. {df.shape[1]} columns.")

# ══════════════════════════════════════════════════════════════
# Cell 6 — Train/Test Split
# ══════════════════════════════════════════════════════════════
TARGET = 'scholarship_label'
NUM_FEATURES = [
    'age', 'year_level', 'gwa_percentage', 'gwa_numeric_1to5',
    'family_annual_income_php', 'family_size',
    'is_solo_parent_dependent', 'is_pwd', 'is_indigenous_people',
    'is_4ps_beneficiary', 'is_ofw_dependent', 'has_existing_scholarship',
    'gwa_fine', 'inc_tier', 'vuln',
    'is_coconut', 'is_sugarcane', 'is_agri', 'is_stem',
    'is_r2', 'is_r6', 'is_dost', 'is_suc', 'is_priv', 'is_luc',
    'sig_univ', 'sig_coll_schol', 'sig_merit_full', 'sig_merit_half',
    'sig_coscho', 'sig_sida', 'sig_broed', 'sig_dost',
    'sig_no_schol', 'sig_tes', 'sig_sikap', 'sig_acef',
    'gwa_x_inc', 'gwa_x_vuln', 'gwa_x_coconut', 'gwa_x_r2',
    'gwa_x_dost', 'gwa_inc_ratio', 'log_income', 'gwa_sq', 'inc_sq',
]
CAT_FEATURES = [
    'sex', 'civil_status', 'region', 'barangay_type',
    'course_category', 'shs_strand', 'enrolled_hei_type',
]

DROP = ['student_id', 'first_name', 'last_name', 'province', 'municipality',
        'parents_occupation', 'high_school_type', 'course']
df.drop(columns=[c for c in DROP if c in df.columns], inplace=True)

X_raw  = df.drop(columns=[TARGET, 'text_profile'])
y      = df[TARGET]
X_text = df['text_profile']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y)
X_train_text, X_test_text = train_test_split(
    X_text, test_size=0.20, random_state=42, stratify=y)
print(f"  Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")

# ══════════════════════════════════════════════════════════════
# Cell 7 — Preprocessing Pipelines
# ══════════════════════════════════════════════════════════════
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipe, NUM_FEATURES),
    ('cat', cat_pipe, CAT_FEATURES),
])

X_train_proc = preprocessor.fit_transform(X_train_raw)
X_test_proc  = preprocessor.transform(X_test_raw)

mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train_proc)
X_test_mm  = mm_scaler.transform(X_test_proc)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2), max_features=500, sublinear_tf=True,
    min_df=2, strip_accents='unicode', analyzer='word',
)
X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
X_test_tfidf  = tfidf.transform(X_test_text).toarray()

X_train_comb = np.hstack([X_train_proc, X_train_tfidf])
X_test_comb  = np.hstack([X_test_proc,  X_test_tfidf])
print(f"  Combined matrix: {X_train_comb.shape}")

# ══════════════════════════════════════════════════════════════
# Cell 8 — Content Profiles
# ══════════════════════════════════════════════════════════════
scholarship_profiles = {}
for label in sorted(y_train.unique()):
    mask = (y_train == label).values
    if mask.sum() > 0:
        scholarship_profiles[label] = X_train_tfidf[mask].mean(axis=0)
print(f"  Built {len(scholarship_profiles)} scholarship profiles.")

# ══════════════════════════════════════════════════════════════
# Cell 9 — Naive Bayes
# ══════════════════════════════════════════════════════════════
print("Training Gaussian NB...")
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train_mm, y_train)
acc_gnb = accuracy_score(y_test, gnb.predict(X_test_mm))
print(f"  GNB Accuracy: {acc_gnb*100:.2f}%")

print("Training Complement NB...")
cnb = ComplementNB(alpha=0.1)
cnb.fit(X_train_tfidf, y_train)
acc_cnb = accuracy_score(y_test, cnb.predict(X_test_tfidf))
print(f"  CNB Accuracy: {acc_cnb*100:.2f}%")

# ══════════════════════════════════════════════════════════════
# Cell 10 — SVM
# ══════════════════════════════════════════════════════════════
print("Training SVM (RBF)...")
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
              class_weight='balanced', random_state=42, decision_function_shape='ovr')
svm_rbf.fit(X_train_comb, y_train)
acc_rbf = accuracy_score(y_test, svm_rbf.predict(X_test_comb))
print(f"  SVM RBF Accuracy: {acc_rbf*100:.2f}%")

print("Training SVM (Linear)...")
svm_lin = SVC(kernel='linear', C=1.0, probability=True,
              class_weight='balanced', random_state=42)
svm_lin.fit(X_train_comb, y_train)
acc_lin = accuracy_score(y_test, svm_lin.predict(X_test_comb))
print(f"  SVM Linear Accuracy: {acc_lin*100:.2f}%")

# ══════════════════════════════════════════════════════════════
# Cell 11 — Random Forest & Gradient Boosting
# ══════════════════════════════════════════════════════════════
print("Training Random Forest (800 trees)...")
rf = RandomForestClassifier(
    n_estimators=800, max_depth=None, min_samples_leaf=1,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_comb, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test_comb))
print(f"  RF Accuracy: {acc_rf*100:.2f}%")

print("Training Gradient Boosting (400 rounds)...")
gb = GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.06, max_depth=6,
    min_samples_leaf=2, subsample=0.85, max_features='sqrt', random_state=42)
gb.fit(X_train_comb, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test_comb))
print(f"  GB Accuracy: {acc_gb*100:.2f}%")

# ══════════════════════════════════════════════════════════════
# Save all .pkl files
# ══════════════════════════════════════════════════════════════
pkl_exports = {
    'model_rf.pkl': rf,
    'model_gb.pkl': gb,
    'model_svm_rbf.pkl': svm_rbf,
    'model_svm_lin.pkl': svm_lin,
    'model_gnb.pkl': gnb,
    'model_cnb.pkl': cnb,
    'preprocessor.pkl': preprocessor,
    'tfidf_vectorizer.pkl': tfidf,
    'mm_scaler.pkl': mm_scaler,
    'content_profiles.pkl': scholarship_profiles,
    'class_labels.pkl': rf.classes_,
}

for filename, obj in pkl_exports.items():
    out_path = os.path.join(PKL_DIR, filename)
    joblib.dump(obj, out_path)

print(f"\nAll .pkl files saved to {PKL_DIR}")
for f in sorted(os.listdir(PKL_DIR)):
    if f.endswith('.pkl'):
        size = os.path.getsize(os.path.join(PKL_DIR, f)) / 1024
        print(f"  {f:<40} {size:6.1f} KB")

print("\nDone! The Django app should now work.")
