import numpy as np
import pandas as pd


DOST_FIELDS = {'Engineering', 'Health Sciences', 'Science', 'IT/Computing', 'Agriculture'}

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


def gwa_fine_tier(x):
    if pd.isna(x):
        return 4
    if x >= 97:
        return 9
    if x >= 95:
        return 8
    if x >= 93:
        return 7
    if x >= 91:
        return 6
    if x >= 88:
        return 5
    if x >= 85:
        return 4
    if x >= 82:
        return 3
    if x >= 78:
        return 2
    return 1


def income_tier(x):
    if pd.isna(x):
        return 3
    if x < 100_000:
        return 1
    if x < 150_000:
        return 2
    if x < 200_000:
        return 3
    if x < 250_000:
        return 4
    if x < 300_000:
        return 5
    if x < 400_000:
        return 6
    if x < 500_000:
        return 7
    return 8


def build_text_profile(row):
    """Build a text profile string from student fields for TF-IDF."""
    if isinstance(row, pd.Series):
        get = lambda k: row.get(k, '')
    else:
        get = lambda k: row.get(k, '')

    parts = []
    parts.append(str(get('course') or '').lower().strip())
    parts.append(str(get('course_category') or '').lower().strip())
    parts.append(str(get('shs_strand') or '').lower().strip())
    parts.append(str(get('region') or '').lower().strip())
    parts.append(str(get('enrolled_hei_type') or '').lower().strip())
    parts.append(str(get('barangay_type') or '').lower().strip())

    occ = str(get('parents_occupation') or '').lower().strip()
    if occ and occ != 'nan':
        parts.append(occ)

    def _flag_val(key):
        v = get(key)
        if v is None:
            return False
        if isinstance(v, (int, float)):
            return v == 1.0
        return str(v).strip().lower() in ('yes', 'y', '1', 'true', 'yep')

    if _flag_val('is_4ps_beneficiary'):
        parts += ['4ps_beneficiary'] * 3
    if _flag_val('is_ofw_dependent'):
        parts += ['ofw_dependent'] * 2
    if _flag_val('is_indigenous_people'):
        parts += ['indigenous_people'] * 2
    if _flag_val('is_pwd'):
        parts += ['pwd'] * 2
    if _flag_val('is_solo_parent_dependent'):
        parts += ['solo_parent'] * 2

    return ' '.join(p for p in parts if p and p != 'nan')


def apply_feature_engineering(input_df):
    """Apply all feature engineering to a DataFrame (can be single-row)."""
    df = input_df.copy()

    g = df['gwa_percentage'].fillna(0)
    inc = df['family_annual_income_php'].fillna(0)

    df['gwa_fine'] = df['gwa_percentage'].apply(gwa_fine_tier)
    df['inc_tier'] = df['family_annual_income_php'].apply(income_tier)

    # Course / field flags
    df['is_coconut'] = (df['course_category'] == 'Coconut-Related').astype(float)
    df['is_sugarcane'] = (df['course_category'] == 'Sugarcane-Related').astype(float)
    df['is_agri'] = df['course_category'].isin(
        {'Agriculture', 'Coconut-Related', 'Sugarcane-Related'}).astype(float)
    df['is_dost'] = df['course_category'].isin(DOST_FIELDS).astype(float)

    # Strand / HEI / Region flags
    df['is_stem'] = (df['shs_strand'] == 'STEM').astype(float)
    df['is_suc'] = (df['enrolled_hei_type'] == 'SUC').astype(float)
    df['is_priv'] = (df['enrolled_hei_type'] == 'Private HEI').astype(float)
    df['is_luc'] = (df['enrolled_hei_type'] == 'LUC').astype(float)
    df['is_r2'] = (df['region'] == 'Region II').astype(float)
    df['is_r6'] = (df['region'] == 'Region VI').astype(float)

    # Vulnerability score
    df['vuln'] = (
        df['is_4ps_beneficiary'].fillna(0) +
        df['is_solo_parent_dependent'].fillna(0) +
        df['is_pwd'].fillna(0) +
        df['is_indigenous_people'].fillna(0) +
        df['is_ofw_dependent'].fillna(0)
    )

    # Scholarship-specific eligibility signals
    df['sig_univ'] = (g >= 97.0).astype(float)
    df['sig_coll_schol'] = ((g >= 95.0) & (g < 97.0)).astype(float)
    df['sig_merit_full'] = ((g >= 96.0) & (g < 99.5)).astype(float)
    df['sig_merit_half'] = ((g >= 93.0) & (g < 95.0)).astype(float)
    df['sig_coscho'] = df['is_coconut']
    df['sig_sida'] = (df['is_sugarcane'] * (inc < 200_000)).astype(float)
    df['sig_broed'] = df['is_r2']
    df['sig_dost'] = (df['is_dost'] * df['is_stem'] * (g >= 85).astype(float))
    df['sig_no_schol'] = (inc > 430_000).astype(float)
    df['sig_tes'] = (
        (inc < 290_000) & df['family_annual_income_php'].notna()
    ).astype(float) * df['vuln'].clip(0, 1)
    df['sig_sikap'] = (inc > 300_000).astype(float) * (1 - df['sig_no_schol'])
    df['sig_acef'] = (
        df['is_agri'] * ((g >= 82) & (g < 95)).astype(float)
    ).astype(float)

    # Interaction features
    df['gwa_x_inc'] = df['gwa_fine'] * df['inc_tier']
    df['gwa_x_vuln'] = df['gwa_fine'] * df['vuln']
    df['gwa_x_coconut'] = g * df['is_coconut']
    df['gwa_x_r2'] = g * df['is_r2']
    df['gwa_x_dost'] = g * df['is_dost']
    df['gwa_inc_ratio'] = df['gwa_fine'] / (df['inc_tier'] + 0.01)

    # Transformation features
    df['log_income'] = np.log1p(df['family_annual_income_php'].fillna(0))
    df['gwa_sq'] = g ** 2
    df['inc_sq'] = df['log_income'] ** 2

    return df


def align_proba(proba, model_classes, target_classes):
    """Reorder probability columns to match a consistent class order."""
    model_classes = list(model_classes)
    aligned = np.zeros((proba.shape[0], len(target_classes)))
    for i, cls in enumerate(target_classes):
        if cls in model_classes:
            aligned[:, i] = proba[:, model_classes.index(cls)]
    return aligned
