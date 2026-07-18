"""
train_model.py
---------------
Trains the supervised ML component of the hybrid diagnosis engine described
in Chapter 3 (3.8 - System Implementation / Classification Module).

Pipeline: TF-IDF vectorization -> Multinomial Naive Bayes / Linear SVM
(both trained; the better cross-validated performer is saved).

Run:  python3 train_model.py
Outputs:
    models/intent_classifier.pkl   (trained sklearn Pipeline)
    models/label_encoder.pkl       (class label list)
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nlp_preprocessor import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "hardware_faults_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["text"].apply(lambda t: preprocess(t, for_ml=True))
    return df


def train():
    df = load_data()
    X = df["clean_text"]
    y = df["fault_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", MultinomialNB(alpha=0.3)),
        ]),
        "linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LinearSVC(C=1.0, random_state=42)),
        ]),
    }

    best_name, best_pipeline, best_score = None, None, -1.0

    print("=" * 60)
    print("MODEL TRAINING & CROSS-VALIDATION")
    print("=" * 60)

    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=4)
        mean_score = scores.mean()
        print(f"\n[{name}] CV accuracy: {mean_score:.3f} (+/- {scores.std():.3f})")

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        print(f"[{name}] Held-out test accuracy: {test_acc:.3f}")
        print(classification_report(y_test, preds, zero_division=0))

        if mean_score > best_score:
            best_name, best_pipeline, best_score = name, pipeline, mean_score

    print("=" * 60)
    print(f"BEST MODEL SELECTED: {best_name} (CV accuracy={best_score:.3f})")
    print("=" * 60)

    # Refit best model on the FULL dataset for production use
    best_pipeline.fit(X, y)

    model_path = os.path.join(MODEL_DIR, "intent_classifier.pkl")
    joblib.dump(best_pipeline, model_path)
    joblib.dump(sorted(y.unique().tolist()), os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(best_name, os.path.join(MODEL_DIR, "model_name.pkl"))

    print(f"\nSaved trained model -> {model_path}")
    return best_pipeline


if __name__ == "__main__":
    train()
