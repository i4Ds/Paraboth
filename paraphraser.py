import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import backoff
import openai
from openai import AzureOpenAI
from tqdm import tqdm

from paraboth.utils import diskcache_cache
from paraboth.paraphraser import BaseParaphraser


class Paraphraser(BaseParaphraser):
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.paraphrasing_not_possible = "paraphrasing_not_possible"
        self.verbose = True
        self.max_retries = 5  # Maximum number of retries on rate limit error
        self.lock = threading.Lock()  # For thread-safe printing

    @diskcache_cache(cache_dir="paraphrase_cache", cache_size=10**9)  # 1GB
    def paraphrase(self, sentence: str, n_sentences: int = 6):
        sentence = sentence.strip()
        if len(sentence) == 0:
            print(f"Error! Received empty sentence for paraphrasing.")
            return f"{self.paraphrasing_not_possible}"
        return self._paraphrase_with_backoff(sentence, n_sentences)

    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_time=60,  # Maximum total retry time in seconds
    )
    def _paraphrase_with_backoff(self, sentence: str, n_sentences: int):
        prompt = f"""
        Paraphrasiere den folgenden Satz <{sentence}> {n_sentences} mal und gib sie folgendermassen zurück: <paraphrased_sentence_1>|<paraphrased_sentence_2>|...|<paraphrased_sentence_n>

        Anweisungen:
        - Behalte die Wortreihenfolge im Satz so weit wie möglich bei.
        - Behalte die ursprüngliche Bedeutung so weit wie möglich bei, während du Variationen einführst.
        - Wenn der Satz im Perfekt steht (z.B. "Er ist schwimmen gegangen"), überlege, ihn umzuformulieren, um der Schweizerdeutschen Syntax zu entsprechen. Beispiel: "er ist schwimmen gegangen" -> "er ging schwimmen".
        - Die Daten, die du erhältst, sind normalisiert, enthalten also keine Sonderzeichen und sind komplett kleingeschrieben.

        Folgende Regeln zählen für den Output:
        - Antworte im normalisierten Format ohne jegliche Satzzeichen.
        - Wenn nichts paraphrasiert werden kann, gibt "{self.paraphrasing_not_possible}" zurück.
        - Falls weniger als {n_sentences} paraphrasierte Sätze generiert werden können, gib nur die generierten Sätze zurück.
        - Es sollen strings in der liste zurückgegeben werden, also <paraphrased_sentence_1>|<paraphrased_sentence_2>|...|<paraphrased_sentence_n>. Sehr wichtig!
        - Es soll IMMER entweder {self.paraphrasing_not_possible} ODER eine Liste mit strings zurückgegeben werden, nichts anderes.

        Deine Ausgabe soll folgendermassen aussehen:
        <paraphrased_sentence_1>|<paraphrased_sentence_2>|...|<paraphrased_sentence_n> 
        ODER
        {self.paraphrasing_not_possible}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Ensure this matches your deployment name
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte im Paraphrasieren von Sätzen.",
                    },
                    {"role": "user", "content": prompt},
                ],
                seed=42,
            )
        except openai.BadRequestError as e:
            print(f"Error: {e}\n")
            return f"{self.paraphrasing_not_possible}"
        # Check for content filtering
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            if self.verbose:
                with self.lock:
                    print(f"Content filtered for sentence: '{sentence}'")
            return f"{self.paraphrasing_not_possible}"
        if choice.message.content is None:
            if self.verbose:
                with self.lock:
                    print(f"No content returned for sentence: '{sentence}'")
                    print(f"Response: {response}")
            return f"{self.paraphrasing_not_possible}"

        answer = response.choices[0].message.content.strip()
        if len(answer) == 0:
            print(f"Error! Paraphrasing '{sentence}' returned an empty string.")
            return f"{self.paraphrasing_not_possible}"

        if self.paraphrasing_not_possible in answer:
            if self.verbose:
                with self.lock:
                    print(f"Paraphrasing not possible for sentence: '{sentence}'")
            return f"{self.paraphrasing_not_possible}"
        return answer

    def paraphrase_list(self, sentences: List[str], n_sentences=6, min_words=3):
        results = [None] * len(sentences)

        def process_sentence(sentence):
            if len(sentence.split()) >= min_words:
                paraphrase = self.paraphrase(sentence, n_sentences)
                if paraphrase != self.paraphrasing_not_possible:
                    paraphrased = [sentence] + paraphrase.split("|")
                else:
                    paraphrased = [sentence]
            else:
                paraphrased = [sentence]
            return [x for x in paraphrased if x.strip() != ""]

        results = [None] * len(sentences)

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {
                executor.submit(process_sentence, sentence): i
                for i, sentence in enumerate(sentences)
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                index = futures[future]
                results[index] = future.result()

        return results


# Usage example
if __name__ == "__main__":
    paraphraser = Paraphraser()
    original_sentences = [
        "Ich habe es versucht",
        "Ich bin müde",
        "Ein Kaffee mit Creme ohne Zucker",
        "Das Velo ist meins.",
    ]
    paraphrased_sentence = paraphraser.paraphrase_list(original_sentences)
    print(f"Original: {original_sentences}")
    print(f"Paraphrased: {paraphrased_sentence}")
