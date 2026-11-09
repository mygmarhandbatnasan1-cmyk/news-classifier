
import re
import unicodedata


def clean_text(text):
    # Unicode хэвийн болгох
    text = unicodedata.normalize("NFC", str(text))

    # Жижиг үсэгт шилжүүлэх
    text = text.lower()

    # HTML тэмдэглэгээ арилгах
    text = re.sub(r"<[^>]+>", " ", text)

    # URL арилгах
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Кирилл, латин үсэг болон тоо үлдээх
    text = re.sub(
        r"[^а-яёөүa-z0-9\s]",
        " ",
        text
    )

    # Илүүдэл зай арилгах
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Туршилт
if __name__  == "__main__":
    sample = (
        "Монголын шигшээ баг 2026 онд "
        "АЛТАН медаль хүртлээ!!!"
    )

    print("Эх текст:", sample)
    print("Цэвэрлэсэн:", clean_text(sample))
