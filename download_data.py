
from pathlib import Path
import pandas as pd
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

URL = (
    "https://raw.githubusercontent.com/"
    "tugstugi/mongolian-nlp/master/datasets/eduge.csv.gz"
)

FILE_PATH = DATA_DIR / "eduge.csv.gz"

LABELS = {
    0: "Урлаг соёл",
    1: "Эдийн засаг",
    2: "Эрүүл мэнд",
    3: "Хууль",
    4: "Улс төр",
    5: "Спорт",
    6: "Технологи",
    7: "Боловсрол",
    8: "Байгаль орчин",
}

# 1. GitHub-аас файл татах
if not FILE_PATH.exists():
    print("Dataset татаж байна...")

    try:
        with requests.get(
            URL, stream=True, timeout=60
        ) as response:
            response.raise_for_status()

            with open(FILE_PATH, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        print("Dataset татагдлаа!")

    except requests.RequestException as e:
        FILE_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"Татах үед алдаа гарлаа: {e}"
        )

# 2. Шахсан CSV файлыг унших
print("CSV файлыг уншиж байна...")

df = pd.read_csv(
    FILE_PATH,
    compression="gzip"
)

print("Баганын нэрс:", df.columns.tolist())
print("Өгөгдлийн хэмжээ:", df.shape)

# 3. Өгөгдлийн эхний мөрүүд
print("\nЭхний 5 мөр:")
print(df.head())

# 4. Боломжтой бол ангиллын нэр нэмэх
if "label" in df.columns:
    df["category"] = pd.to_numeric(
        df["label"], errors="coerce"
    ).map(LABELS)

# 5. CSV хадгалах
output = DATA_DIR / "all_news.csv"

df.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\nХадгалсан файл:", output)
print("Нийт мөр:", len(df))

if "category" in df.columns:
    print("\nАнгиллын статистик:")
    print(df["category"].value_counts())

print("\nАмжилттай дууслаа!")
data/explore_data.py