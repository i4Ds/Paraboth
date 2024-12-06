import re

import nltk


class Text:
    def __init__(self, file_path=None, content=None, split_delimiters=","):
        self.file_path = file_path
        self.sentences = []
        self.raw_text = ""
        self.split_delimiters = split_delimiters
        # 1. Read the file
        if content is None:
            with open(self.file_path, "r", encoding="utf-8") as file:
                self.content = file.read()
        else:
            self.content = content
        # Necessary normalization
        self._normalization()
        self.file_type = self._detect_file_type()
        print(f"Detected File type: {self.file_type}")
        self._process_content()

    def _normalization(self):
        self.content = re.sub(
            r"(\d+),(\d+)", r"\1.\2", self.content
        )  # replace commas with dots in numbers
        if (
            "Sprecher Identifikation\n" in self.content
            and "\nTranskript\n" in self.content
        ):
            self.content = self.content.split("\nTranskript\n")[1]

    def _detect_file_type(self):
        """Detects the file type based on specific patterns in the first few lines of the content."""
        lines = self.content.split("\n")
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
            # Check for 'Speaker X | timestamp' pattern
            if re.match(r"Speaker\s+\d+\s*\|\s*\d+:\d+\.\d+", line):
                return "speaker_timestamp"
            # Existing checks
            if len(line) > 1 and line[1] == ":":
                return "speaker_colon"
            elif "Sprecher" in line:
                return "timestamp"
        return "default"

    def _process_content(self):
        """Processes the content into sentences based on detected file type."""
        if self.file_type == "timestamp":
            self._process_timestamp_format(self.content)
        elif self.file_type == "speaker_colon":
            self._process_speaker_colon_format(self.content)
        elif self.file_type == "speaker_timestamp":
            self._process_speaker_timestamp_format(self.content)
        else:
            self._process_default_format(self.content)

    def _process_timestamp_format(self, content):
        """Processes content with timestamp and speaker format."""
        segments = re.split(r"\[Sprecher \d+ \d+:\d+\]", content)
        segments = [seg.strip() for seg in segments if seg.strip()]
        for segment in segments:
            self._process_line(segment)

    def _process_speaker_colon_format(self, content):
        """Processes content with speaker: text format."""
        lines = content.split("\n")
        for line in lines:
            if ":" in line:
                _, text = line.split(":", 1)
                text = text.strip()
            else:
                text = line.strip()
            # Ensure the sentence ends with proper punctuation
            if not text.endswith((".", "!", "?", ",")):
                text += "."
            self._process_line(text)

    def _process_speaker_timestamp_format(self, content: str):
        """Processes content with speaker and timestamp format."""
        lines = content.split("\n")
        for line in lines:
            if "Speaker" in line:
                # Skip speaker lines
                continue
            else:
                self._process_line(line.strip())

    def _process_default_format(self, content):
        """Processes content without specific speaker information."""
        self._process_line(content)

    def _process_line(self, text: str):
        """Splits text into sentences and adds them to the sentences list."""
        text = text.strip()
        if text:
            self.raw_text += " " + text

    def to_list_of_strings(self):
        """Returns the processed sentences as a list of strings."""
        sentences = nltk.sent_tokenize(self.raw_text, language="german")
        # Split by split_delimiters
        split_sentences = []
        for sentence in sentences:
            split_sentences.extend(
                re.split(f"[{re.escape(self.split_delimiters)}]", sentence)
            )
        self.sentences = [
            sentence.strip() for sentence in split_sentences if sentence.strip()
        ]
        return self.sentences

    def to_string(self):
        """Returns the processed sentences as a single string."""
        return self.raw_text

    def get_file_type(self):
        """Returns the detected file type."""
        return self.file_type


if __name__ == "__main__":
    from paraboth.normalizer import TextNormalizer

    n = TextNormalizer()
    file_path = "data/09_gt.txt"  # Replace with your actual file path
    text = Text(file_path)
    print("Detected File Type:", text.get_file_type())
    print("\nProcessed Sentences:")
    for sentence in text.to_list_of_strings():
        print(sentence)
