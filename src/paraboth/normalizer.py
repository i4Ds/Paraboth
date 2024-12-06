import re
from typing import List


class TextNormalizer:
    def __init__(
        self,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        replace_umlauts: bool = True,
    ):
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.replace_umlauts = replace_umlauts

    def normalize(self, texts: List[str]) -> List[str]:
        normalized = [self._normalize_text(text) for text in texts if text]
        return [x for x in normalized if x is not None]

    def _normalize_text(self, text: str) -> str:
        if self.replace_umlauts:
            text = self._replace_umlauts(text)

        if self.remove_punctuation:
            # This replaces commas that are between two digits
            text = re.sub(r"(?<=\d),(?=\d)", "", text)
            text = re.sub(r"\.(?!\d)", "", text)
            text = re.sub(r"[^\w\s.]|_", "", text)

        if self.replace_umlauts:
            text = self._replace_umlauts(text)

        if self.lowercase:
            text = text.lower()

        text = text.strip()

        if text is None or len(text) == 0:
            return None

        text = re.sub(
            r"\s+", " ", text
        )  # replace any successive whitespace characters with a space

        return text.strip()

    @staticmethod
    def _replace_umlauts(text: str) -> str:
        umlaut_map = {
            "ä": "ae",
            "ö": "oe",
            "ü": "ue",
            "Ä": "Ae",
            "Ö": "Oe",
            "Ü": "Ue",
            "ß": "ss",
        }
        for umlaut, replacement in umlaut_map.items():
            text = text.replace(umlaut, replacement)
        return text


# Example usage
if __name__ == "__main__":
    texts = [
        "Hallo, wie geht's dir?",
        "Ich bin 25 Jahre alt und wohne in München.",
        "Das Wetter ist schön, aber es ist 30°C draußen!",
        "Ich nehme täglich 2.5mg Aspirin ein.",
    ]

    normalizer = TextNormalizer(
        remove_punctuation=True, lowercase=True, replace_umlauts=True
    )

    normalized_texts = normalizer.normalize(texts)
    for original, normalized in zip(texts, normalized_texts):
        print(f"Original: {original}")
        print(f"Normalized: {normalized}")
        print()
