import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from features import extract_mel

DATASET_PATH = "dataset"


def load_data(split):
    X, y = [], []
    for label, cls in enumerate(["normal", "abnormal"]):
        folder = os.path.join(DATASET_PATH, split, cls)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                try:
                    path = os.path.join(folder, file)
                    X.append(extract_mel(path).squeeze().flatten())
                    y.append(label)
                except Exception as e:
                    print(f"Atlandi {path}: {e}")
    return np.array(X), np.array(y)


def find_threshold(clf, X_val, y_val, min_abnormal_recall=0.45):
    """Anormal yakalamayi yeterli tutup normali de koruyan esik sec."""
    probs = clf.predict_proba(X_val)[:, 1]
    best_t, best_no = 0.5, -1.0
    best_score = -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (probs >= t).astype(int)
        ab_recall = np.mean(pred[y_val == 1] == 1) if np.any(y_val == 1) else 0.0
        no_recall = np.mean(pred[y_val == 0] == 0) if np.any(y_val == 0) else 0.0
        if ab_recall < min_abnormal_recall:
            continue
        if no_recall > best_no:
            best_no, best_t = no_recall, t
            best_score = 0.6 * ab_recall + 0.4 * no_recall
    if best_no < 0:
        # Yedek: dengeli skor
        for t in np.linspace(0.05, 0.95, 181):
            pred = (probs >= t).astype(int)
            ab_recall = np.mean(pred[y_val == 1] == 1) if np.any(y_val == 1) else 0.0
            no_recall = np.mean(pred[y_val == 0] == 0) if np.any(y_val == 0) else 0.0
            score = 0.6 * ab_recall + 0.4 * no_recall
            if score > best_score:
                best_score, best_t = score, t
    return float(best_t), float(best_score)


print("Loading dataset...")
X_train, y_train = load_data("train")
X_val, y_val = load_data("validation")
X_test, y_test = load_data("test")

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train distribution:", np.unique(y_train, return_counts=True))

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

threshold, val_score = find_threshold(clf, X_val, y_val)
print(f"Validation esik: {threshold:.3f} (dengeli skor={val_score:.3f})")

probs_test = clf.predict_proba(X_test)[:, 1]
y_pred = (probs_test >= threshold).astype(int)

print("\nTest raporu:")
print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))

ab_recall = float(np.mean(y_pred[y_test == 1] == 1)) if np.any(y_test == 1) else 0.0
no_recall = float(np.mean(y_pred[y_test == 0] == 0)) if np.any(y_test == 0) else 0.0
test_acc = float(np.mean(y_pred == y_test))

bundle = {"model": clf, "threshold": threshold}
joblib.dump(bundle, "model.joblib")

meta = {
    "test_accuracy": test_acc,
    "abnormal_recall": ab_recall,
    "normal_recall": no_recall,
    "threshold": threshold,
    "model_type": "random_forest",
}
with open("model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Kaydedildi: model.joblib")
print(f"Test abnormal recall: {ab_recall:.3f}, normal recall: {no_recall:.3f}")
