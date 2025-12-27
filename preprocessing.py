import re

def clean_arabic(text):
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)  # keep Arabic only
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize Arabic letters
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)

    return text
