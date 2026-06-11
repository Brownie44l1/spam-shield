import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/sms_spam.tsv")


def train():
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df.dropna(inplace=True)

    print(f"   Total samples : {len(df)}")
    print(f"   Spam          : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"   Ham           : {(df['label'] == 0).sum()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Pipeline: TF-IDF vectorizer → Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z0-9]+\b",
            ngram_range=(1, 2),     # unigrams + bigrams
            max_df=0.95,
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    print("🏋️  Training model...")
    pipeline.fit(X_train, y_train)

    print("📊 Evaluating...\n")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy : {acc*100:.2f}%")
    print()
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("   Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}  TP={cm[1,1]}\n")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")
    return pipeline


def load():
    if not os.path.exists(MODEL_PATH):
        print("⚙️  No saved model found — training now...")
        return train()
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train()
