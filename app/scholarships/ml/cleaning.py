import re
import numpy as np
import pandas as pd


YES_VALS = {'yes', 'y', '1', 'true', 'yep'}
NO_VALS = {'no', 'n', '0', 'false', 'nope', 'nop'}


def clean_binary(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in YES_VALS:
        return 1.0
    if s in NO_VALS:
        return 0.0
    return np.nan


def clean_sex(v):
    if pd.isna(v):
        return 'Unknown'
    s = str(v).strip().lower()
    if s.startswith('f'):
        return 'Female'
    if s.startswith('m'):
        return 'Male'
    return 'Unknown'


def clean_civil(v):
    if pd.isna(v):
        return 'Unknown'
    mapping = {
        'single': 'Single', 'married': 'Married', 'widowed': 'Widowed',
        'singe': 'Single', 'sngle': 'Single',
    }
    return mapping.get(str(v).strip().lower(), 'Unknown')


def clean_hei(v):
    if pd.isna(v):
        return 'Unknown'
    s = re.sub(r'[\s._-]+', '', str(v).strip().lower())
    if 'suc' in s or 'stateuniversity' in s:
        return 'SUC'
    if 'luc' in s or 'localuniversity' in s:
        return 'LUC'
    if 'private' in s:
        return 'Private HEI'
    return 'Unknown'


def clean_strand(v):
    if pd.isna(v):
        return 'Unknown'
    s = str(v).strip().lower()
    if 'stem' in s:
        return 'STEM'
    if 'abm' in s and 'tvl' in s:
        return 'TVL/ABM'
    if 'abm' in s:
        return 'ABM'
    if 'humss' in s or 'gas' in s or 'academic' in s:
        return 'HUMSS/GAS'
    if 'tvl' in s:
        return 'TVL'
    return 'Other'


def clean_bar(v):
    if pd.isna(v):
        return 'Unknown'
    s = str(v).strip().lower()
    if 'urban' in s:
        return 'Urban'
    if 'rural' in s:
        return 'Rural'
    return 'Unknown'


def clean_cat(v):
    if pd.isna(v):
        return 'Unknown'
    s = str(v).strip().lower()
    MAP = {
        'coconut-related': 'Coconut-Related',
        'sugarcane-related': 'Sugarcane-Related',
        'engineering': 'Engineering',
        'enginnering': 'Engineering',
        'agriculture': 'Agriculture',
        'agriulture': 'Agriculture',
        'health sciences': 'Health Sciences',
        'health sciencess': 'Health Sciences',
        'health science': 'Health Sciences',
        'helath sciences': 'Health Sciences',
        'science': 'Science',
        'sience': 'Science',
        'sciience': 'Science',
        'social sciences': 'Social Sciences',
        'hospitality': 'Hospitality',
        'it/computing': 'IT/Computing',
        'i.t/computing': 'IT/Computing',
        'business': 'Business',
        'bussiness': 'Business',
        'buisness': 'Business',
        'busines': 'Business',
        'education': 'Education',
    }
    for k, val in MAP.items():
        if k in s:
            return val
    return s.title()


def clean_region(v):
    if pd.isna(v):
        return 'Unknown'
    return re.sub(r'\s+', ' ', str(v).strip())


BINARY_COLS = [
    'is_solo_parent_dependent', 'is_pwd', 'is_indigenous_people',
    'is_4ps_beneficiary', 'is_ofw_dependent', 'has_existing_scholarship',
]

NUMERIC_COLS = [
    'gwa_percentage', 'gwa_numeric_1to5',
    'family_annual_income_php', 'family_size', 'age', 'year_level',
]


def clean_student_input(data):
    """Clean a student dict, applying all normalization functions."""
    cleaned = dict(data)

    if 'sex' in cleaned:
        cleaned['sex'] = clean_sex(cleaned['sex'])
    if 'civil_status' in cleaned:
        cleaned['civil_status'] = clean_civil(cleaned['civil_status'])
    if 'enrolled_hei_type' in cleaned:
        cleaned['enrolled_hei_type'] = clean_hei(cleaned['enrolled_hei_type'])
    if 'shs_strand' in cleaned:
        cleaned['shs_strand'] = clean_strand(cleaned['shs_strand'])
    if 'barangay_type' in cleaned:
        cleaned['barangay_type'] = clean_bar(cleaned['barangay_type'])
    if 'course_category' in cleaned:
        cleaned['course_category'] = clean_cat(cleaned['course_category'])
    if 'region' in cleaned:
        cleaned['region'] = clean_region(cleaned['region'])

    for col in BINARY_COLS:
        if col in cleaned:
            val = cleaned[col]
            if isinstance(val, bool):
                cleaned[col] = 1.0 if val else 0.0
            else:
                cleaned[col] = clean_binary(val)

    for col in NUMERIC_COLS:
        if col in cleaned:
            try:
                cleaned[col] = float(cleaned[col]) if cleaned[col] is not None else np.nan
            except (ValueError, TypeError):
                cleaned[col] = np.nan

    return cleaned
