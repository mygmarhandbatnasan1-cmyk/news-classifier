
import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("data")

# Өгөгдөл унших
df = pd.read_csv(DATA_DIR / "all_news.csv")

# Баганын нэрний илүүдэл зайг арилгах
df.columns = df.columns.str.strip()

print("Эхний мэдээний тоо:", len(df))

# Хоосон утга арилгах
df = df.dropna(subset=["news", "label"])

# Давхардсан мэдээ арилгах
df = df.drop_duplicates(subset=["news"]).copy()

# Текст цэвэрлэх функц
def clean_text(text):
    text = str(text).lower()

    # HTML таг арилгах
    text = re.sub(r"<[^>]+>", " ", text)

    # URL арилгах
    text = re.sub(r"https?://\S+", " ", text)

    # Кирилл, латин үсэг, тоо, зай үлдээх
    text = re.sub(
        r"[^а-яёөүa-z0-9\s]",
        " ",
        text
    )

    # Илүүдэл зай арилгах
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Текст цэвэрлэх
df["clean_text"] = df["news"].apply(clean_text)

# Цэвэрлэгээний дараа хоосон болсон мөр хасах
df = df[df["clean_text"].str.len() > 0].copy()

# Цэвэрлэсэн текстийн давхардал арилгах
df = df.drop_duplicates(subset=["clean_text"])

# Үгийн тоо
df["word_count"] = df["clean_text"].str.split().str.len()

# Хадгалах
output = DATA_DIR / "cleaned_news.csv"

df.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\nЦэвэрлэсний дараах мэдээ:", len(df))
print("Хасагдсан мэдээ:", 75661 - len(df))
print("Хадгалсан файл:", output)

print("\nАнгиллын тоо:")
print(df["label"].value_counts())

print("\nЦэвэрлэсэн текстийн жишээ:")
print(df["clean_text"].iloc[0][:500])
