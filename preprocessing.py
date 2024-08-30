import re


def preprocessing(batch_text):
    text = batch_text
    text = text.lower()  # Lowercase the text.
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces.
    text = text.strip()  # Remove leading and trailing whitespaces.
    return text
