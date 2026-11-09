
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Хавтас бэлтгэх
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# 2. Цэвэрлэсэн өгөгдөл унших
df = pd.read_csv(DATA_DIR / "cleaned_news.csv")
df.columns = df.columns.str.strip()

df = df.dropna(subset=["clean_text", "label"])
df["clean_text"] = df["clean_text"].astype(str)
df["label"] = df["label"].astype(str).str.strip()

# 3. Давхардсан цэвэр текстийг шалгах
df = df.drop_duplicates(subset=["clean_text"])

X = df["clean_text"]
y = df["label"]

print("Нийт мэдээ:", len(df))
print("Ангиллын тоо:", y.nunique())

# 4. Train 70%, Validation 15%, Test 15%
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

print("\n ӨГӨГДЛИЙН ХУВААЛТ ")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# 5. TF-IDF + Naive Bayes Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )),
    ("classifier", MultinomialNB())
])

# 6. Загвар сургах
print("\nЗагвар сургаж байна...")
model.fit(X_train, y_train)
print("Загвар сургаж дууслаа!")

# 7. Validation дээр үнэлэх
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(
    y_val, y_pred, average="macro"
)

print("\n VALIDATION ҮР ДҮН ")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\n АНГИЛАЛ БҮРИЙН ҮР ДҮН ")
report = classification_report(
    y_val,
    y_pred,
    zero_division=0
)
print(report)

# 8. Үр дүнг хадгалах
with open(
    RESULT_DIR / "naive_bayes_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)
    f.write(f"\nAccuracy: {accuracy:.4f}")
    f.write(f"\nMacro F1: {macro_f1:.4f}")

# 9. Загвараа хадгалах
joblib.dump(
    model,
    MODEL_DIR / "naive_bayes.joblib"
)

# 10. Validation болон Test өгөгдөл хадгалах
pd.DataFrame({
    "text": X_val,
    "label": y_val
}).to_csv(
    DATA_DIR / "validation.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "text": X_test,
    "label": y_test
}).to_csv(
    DATA_DIR / "test.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nЗагвар болон үр дүн хадгалагдлаа!")
