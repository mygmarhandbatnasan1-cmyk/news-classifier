
import joblib
import re

# 1. Сурсан Linear SVM загварыг ачаалах
model = joblib.load("models/linear_svm.joblib")


# 2. Текст цэвэрлэх
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(
        r"[^а-яёөүa-z0-9\s]",
        " ",
        text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 3. Мэдээ ангилах функц
def classify_news(text):
    cleaned = clean_text(text)

    if not cleaned:
        return "Ангилах боломжгүй"

    prediction = model.predict([cleaned])[0]

    return prediction


# 4. Хэрэглэгчээс мэдээ авах
while True:
    print("\n МОНГОЛ МЭДЭЭ АНГИЛАГЧ ")

    news = input(
        "Мэдээний текстээ бичнэ үү (гарах бол exit): "
    )

    if news.lower() == "exit":
        break

    category = classify_news(news)

    print("\nТаамагласан сэдэв:", category)
