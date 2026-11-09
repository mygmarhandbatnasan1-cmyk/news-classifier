
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

# 1. Үр дүн хадгалах хавтас
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# 2. Хамгийн сайн Linear SVM загварыг унших
model = joblib.load("models/linear_svm.joblib")

# 3. Validation өгөгдөл унших
df = pd.read_csv("data/validation.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["text", "label"])

X = df["text"].astype(str)
y_true = df["label"].astype(str).str.strip()

# 4. Таамаглал гаргах
y_pred = model.predict(X)

print(" LINEAR SVM ҮР ДҮН ")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro F1:", f1_score(
    y_true, y_pred, average="macro"
))

print("\n CLASSIFICATION REPORT ")
print(classification_report(
    y_true, y_pred, zero_division=0
))

# 5. Confusion Matrix
labels = sorted(model.classes_)

cm = confusion_matrix(
    y_true, y_pred, labels=labels
)

# Мөр тус бүрээр хувьд шилжүүлэх
cm_percent = (
    cm / cm.sum(axis=1, keepdims=True) * 100
)

plt.figure(figsize=(13, 10))

sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=range(1, len(labels) + 1),
    yticklabels=range(1, len(labels) + 1)
)

plt.title("Linear SVM - Confusion Matrix (%)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 6. Ангиллын нэрсийн тайлбар
print("\n АНГИЛЛЫН ДУГААР ")
for i, label in enumerate(labels, 1):
    print(f"{i}. {label}")

# 7. Алдаатай ангилсан мэдээнүүд
errors = pd.DataFrame({
    "text": X.values,
    "actual": y_true.values,
    "predicted": y_pred
})

errors = errors[
    errors["actual"] != errors["predicted"]
].copy()

errors.to_csv(
    RESULT_DIR / "misclassified_news.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nНийт алдаатай ангилсан мэдээ:", len(errors))
print("\nХамгийн эхний 5 алдаа:")
print(errors.head().to_string(index=False))

print("\nҮр дүн results хавтсанд хадгалагдлаа!")
