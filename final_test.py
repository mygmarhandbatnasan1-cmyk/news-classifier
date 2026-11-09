
from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Хавтас
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# 2. Өмнө сургасан шилдэг загвар
model = joblib.load("models/linear_svm.joblib")

# 3. Өмнө хадгалсан TEST өгөгдөл
df = pd.read_csv("data/test.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["text", "label"])

X_test = df["text"].astype(str)
y_test = df["label"].astype(str).str.strip()

print("Test мэдээний тоо:", len(df))

# 4. Автомат ангилалт
y_pred = model.predict(X_test)

# 5. Үнэлгээ
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="macro",
    zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="macro",
    zero_division=0
)
macro_f1 = f1_score(
    y_test, y_pred, average="macro"
)

report = classification_report(
    y_test, y_pred, zero_division=0
)

print("\n=== FINAL TEST RESULT ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(report)

# 6. Үр дүн хадгалах
output = RESULT_DIR / "final_test_report.txt"

with open(output, "w", encoding="utf-8") as f:
    f.write("FINAL TEST RESULT\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write(report)

print("\nХадгаллаа:", output)
