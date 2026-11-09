
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# 1. Үр дүн хадгалах хавтас
Path("results").mkdir(exist_ok=True)

# 2. Цэвэрлэсэн өгөгдөл унших
df = pd.read_csv("data/cleaned_news.csv")
df.columns = df.columns.str.strip()

df = df.dropna(subset=["clean_text", "label"])
df["clean_text"] = df["clean_text"].astype(str)
df["label"] = df["label"].astype(str).str.strip()
df = df.drop_duplicates(subset=["clean_text"])

X = df["clean_text"]
y = df["label"]

print("Нийт мэдээ:", len(df))

# 3. Өмнөх туршилттай ижил хуваалт
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

# 4. N-gram тохиргоонууд
experiments = {
    "Unigram": (1, 1),
    "Bigram": (2, 2),
    "Trigram": (3, 3),
    "Unigram + Bigram": (1, 2),
    "Unigram + Bigram + Trigram": (1, 3)
}

results = []

# 5. Туршилт бүрийг ажиллуулах
for name, ngram in experiments.items():

    print(f"\n=== {name} ===")
    print("N-gram:", ngram)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram,
            max_features=50000,
            min_df=3,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("classifier", LinearSVC(
            random_state=42
        ))
    ])

    print("Загвар сургаж байна...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    feature_count = len(
        model.named_steps["tfidf"].get_feature_names_out()
    )

    print("Шинжийн тоо:", feature_count)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    results.append({
        "Experiment": name,
        "Ngram": str(ngram),
        "Features": feature_count,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1
    })

# 6. Харьцуулалтын хүснэгт
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\n=== N-GRAM ХАРЬЦУУЛАЛТ ===")
print(results_df.to_string(index=False))

# 7. CSV хадгалах
results_df.to_csv(
    "results/ngram_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# 8. График үүсгэх
plot_df = results_df.sort_values(
    "Macro_F1",
    ascending=True
)

plt.figure(figsize=(11, 6))

plt.barh(
    plot_df["Experiment"],
    plot_df["Macro_F1"] * 100
)

plt.xlabel("Macro F1-score (%)")
plt.ylabel("N-gram method")
plt.title("N-gram Comparison - Linear SVM")
plt.xlim(0, 100)
plt.tight_layout()

plt.savefig(
    "results/ngram_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 9. Хамгийн сайн тохиргоо
best = results_df.iloc[0]

print("\n ХАМГИЙН САЙН N-GRAM ")
print("Тохиргоо:", best["Experiment"])
print(f"Accuracy: {best['Accuracy']:.4f}")
print(f"Macro F1: {best['Macro_F1']:.4f}")

print("\nТуршилт амжилттай дууслаа!")
