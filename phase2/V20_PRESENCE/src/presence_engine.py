import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def v20_a_baseline(features):
    return features["max_score"]

def v20_b_multi_evidence(features):
    s_term = np.clip(features["max_score"], 0.0, 1.0)
    psr_term = np.clip(features["psr"] / 12.0, 0.0, 1.0)
    context_term = np.clip(features["context_score"], 0.0, 1.0)
    phase_term = np.clip(features["phase_residual"], 0.0, 1.0)
    
    return float(0.25 * s_term + 0.25 * psr_term + 0.30 * context_term + 0.20 * phase_term)

def train_v20_d_classifier(X_train, y_train):
    clf = LogisticRegression(class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    return clf

def predict_v20_d_classifier(clf, X):
    return clf.predict_proba(X)[:, 1]

def v20_e_select_threshold(y_prob, y_true):
    thresholds = np.linspace(0, 1, 101)
    best_f1 = -1
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh
