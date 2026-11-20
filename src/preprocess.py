import re

def clean_text(text: str) -> str:
    text = text.lower()                                   # "URGENT" -> "urgent"
    text = re.sub(r"[^a-z0-9\s]", " ", text)              # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()              # collapse double spaces
    return text