
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# 1. Хавтас бэлтгэх
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# 2. Өгөгдөл унших
df = pd.read_csv("data/cleaned_news.csv")
df.columns = df.columns.str.strip()

df = df.dropna(subset=["clean_text", "label"])
df["clean_text"] = df["clean_text"].astype(str)
df["label"] = df["label"].astype(str).str.strip()
df = df.drop_duplicates(subset=["clean_text"])

X = df["clean_text"]
y = df["label"]

# 3. Өмнөхтэй ИЖИЛ хуваалт
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# 4. Харьцуулах алгоритмууд
algorithms = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Linear SVM": LinearSVC(
        random_state=42
    )
}

results = []

# 5. Загвар бүрийг сургах
for name, classifier in algorithms.items():

    print(f"\n {name} ")
    print("Сургаж байна...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    # Зөвхөн validation дээр үнэлэх
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print(classification_report(
        y_val,
        predictions,
        zero_division=0
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1
    })

    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        f"models/{filename}.joblib"
    )

# 6. Үр дүнг харьцуулах
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\n ЗАГВАРУУДЫН ХАРЬЦУУЛАЛТ ")
print(results_df.to_string(index=False))

# 7. Үр дүн хадгалах
results_df.to_csv(
    "results/model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nХамгийн сайн загвар:")
print(results_df.iloc[0]["Model"])

print("\nХарьцуулалт амжилттай дууслаа!")
