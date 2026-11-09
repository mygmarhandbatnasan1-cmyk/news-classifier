
import pandas as pd

df = pd.read_csv("data/all_news.csv")
df.columns = df.columns.str.strip()

print("=== 1. ӨГӨГДЛИЙН ХЭМЖЭЭ ===")
print(df.shape)

print("\n=== 2. БАГАНЫН НЭР ===")
print(df.columns.tolist())

print("\n=== 3. АНГИЛЛЫН ТОО ===")
print(df["label"].value_counts())

print("\n=== 4. ХООСОН УТГА ===")
print(df.isnull().sum())

print("\n=== 5. ДАВХАРДСАН МЭДЭЭ ===")
print(df.duplicated(subset=["news"]).sum())

print("\n=== 6. МЭДЭЭНИЙ ҮГИЙН ТОО ===")
df["word_count"] = (
    df["news"]
    .fillna("")
    .astype(str)
    .str.split()
    .str.len()
)

print(df["word_count"].describe())

print("\n=== 7. АНГИЛАЛ БҮРИЙН ДУНДАЖ УРТ ===")
print(df.groupby("label")["word_count"].mean())
