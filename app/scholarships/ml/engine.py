import os
import threading
import logging
from pathlib import Path

os.environ.setdefault('JOBLIB_MULTIPROCESSING', '0')

import joblib
import numpy as np
import pandas as pd

from . import cleaning, features

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_models = {}
_is_loaded = False

PKL_DIR = Path(__file__).resolve().parent.parent / 'pkl'

ENSEMBLE_WEIGHTS = {
    'rf': 0.30,
    'gb': 0.25,
    'svm_rbf': 0.25,
    'gnb': 0.10,
    'cnb': 0.10,
}

REQUIRED_PKL_FILES = {
    'preprocessor': 'preprocessor.pkl',
    'tfidf': 'tfidf_vectorizer.pkl',
    'mm_scaler': 'mm_scaler.pkl',
    'class_labels': 'class_labels.pkl',
}

MODEL_PKL_FILES = {
    'rf': 'model_rf.pkl',
    'gb': 'model_gb.pkl',
    'svm_rbf': 'model_svm_rbf.pkl',
    'svm_lin': 'model_svm_lin.pkl',
    'gnb': 'model_gnb.pkl',
    'cnb': 'model_cnb.pkl',
}

OPTIONAL_PKL_FILES = {
    'content_profiles': 'content_profiles.pkl',
}


def load_models():
    """Load all pkl artifacts into memory. Thread-safe, idempotent."""
    global _models, _is_loaded
    if _is_loaded:
        return
    with _lock:
        if _is_loaded:
            return

        for key, filename in REQUIRED_PKL_FILES.items():
            path = PKL_DIR / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}. "
                    f"Run the notebook first to generate .pkl files, "
                    f"then copy them to {PKL_DIR}"
                )
            _models[key] = joblib.load(path)
            logger.info("Loaded %s from %s", key, path)

        loaded_weighted_model = False
        for key, filename in MODEL_PKL_FILES.items():
            path = PKL_DIR / filename
            if not path.exists():
                logger.warning("Optional model file not found: %s", path)
                continue
            _models[key] = joblib.load(path)
            loaded_weighted_model = loaded_weighted_model or key in ENSEMBLE_WEIGHTS
            logger.info("Loaded %s from %s", key, path)

        if not loaded_weighted_model:
            raise FileNotFoundError(f"No inference model files found in {PKL_DIR}")

        for key, filename in OPTIONAL_PKL_FILES.items():
            path = PKL_DIR / filename
            if path.exists():
                _models[key] = joblib.load(path)
                logger.info("Loaded %s from %s", key, path)

        _is_loaded = True
        logger.info("All ML models loaded successfully.")


def recommend_scholarship(student_dict, top_n=3):
    """
    Full inference pipeline: raw student dict -> top N recommendations.

    Parameters
    ----------
    student_dict : dict
        Raw student profile with fields matching the dataset columns.
    top_n : int
        Number of top recommendations to return.

    Returns
    -------
    list of dict
        Each dict has keys: 'scholarship', 'confidence', 'rank'
    """
    load_models()

    # 1. Clean raw input
    cleaned = cleaning.clean_student_input(student_dict)

    # 2. Build DataFrame and apply feature engineering
    input_df = pd.DataFrame([cleaned])

    # Cross-fill GWA if needed
    if pd.isna(input_df['gwa_percentage'].iloc[0]) and not pd.isna(input_df.get('gwa_numeric_1to5', pd.Series([np.nan])).iloc[0]):
        input_df['gwa_percentage'] = 100 - (input_df['gwa_numeric_1to5'] - 1) * 7.5
    if pd.isna(input_df.get('gwa_numeric_1to5', pd.Series([np.nan])).iloc[0]) and not pd.isna(input_df['gwa_percentage'].iloc[0]):
        input_df['gwa_numeric_1to5'] = (100 - input_df['gwa_percentage']) / 7.5 + 1

    input_df = features.apply_feature_engineering(input_df)

    # 3. Build text profile
    text_profile = features.build_text_profile(student_dict)

    # 4. Transform through preprocessor, scaler, TF-IDF
    X_num = _models['preprocessor'].transform(input_df)
    X_mm = _models['mm_scaler'].transform(X_num)
    X_tfidf = _models['tfidf'].transform([text_profile]).toarray()
    X_comb = np.hstack([X_num, X_tfidf])

    # 5. Get aligned probabilities from available models
    classes_ = _models['class_labels']
    weighted_probs = []
    active_weight = 0.0

    model_inputs = {
        'rf': X_comb,
        'gb': X_comb,
        'svm_rbf': X_comb,
        'gnb': X_mm,
        'cnb': X_tfidf,
    }

    for key, model_input in model_inputs.items():
        model = _models.get(key)
        if model is None:
            continue
        weight = ENSEMBLE_WEIGHTS[key]
        proba = features.align_proba(
            model.predict_proba(model_input),
            model.classes_,
            classes_,
        )
        weighted_probs.append(weight * proba)
        active_weight += weight

    if not weighted_probs or active_weight == 0:
        raise RuntimeError("No loaded inference models are available.")

    # 6. Weighted ensemble, renormalized when optional models are excluded.
    ens_prob = sum(weighted_probs) / active_weight

    top_idx = ens_prob[0].argsort()[::-1][:top_n]

    return [
        {
            'scholarship': str(classes_[i]),
            'confidence': float(ens_prob[0, i]) * 100,
            'rank': rank,
        }
        for rank, i in enumerate(top_idx, 1)
    ]
