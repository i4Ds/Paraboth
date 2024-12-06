import os

import numpy as np
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from paraboth.utils import diskcache_cache

class BaseEmbedder:
    """
    An abstract base class for text embedding operations.

    This class defines the interface for embedding-related operations such as
    generating embeddings for text chunks, retrieving embeddings, and saving/loading
    embeddings to/from files. Concrete implementations should inherit from this class
    and provide specific implementations for these methods.
    Methods
    -------
    embed_chunks(text_chunks, batch_size=100)
        Abstract method to embed a list of text chunks.
    get_embeddings()
        Abstract method to retrieve stored embeddings.
    save_embeddings(file_path)
        Abstract method to save embeddings to a file.
    load_embeddings(file_path)
        Abstract method to load embeddings from a file.
    """

    def embed_chunks(self, text_chunks, batch_size=100):
        raise NotImplementedError

    def get_embeddings(self):
        raise NotImplementedError

    def save_embeddings(self, file_path):
        raise NotImplementedError

    def load_embeddings(self, file_path):
        raise NotImplementedError

class Embedder(BaseEmbedder):
    """
    A class for generating and managing text embeddings using Azure OpenAI.

    This class provides functionality to embed text chunks into vector representations
    using the Azure OpenAI API. It also includes methods for storing, retrieving,
    saving, and loading embeddings.

    Attributes
    ----------
    client : AzureOpenAI
        The Azure OpenAI client for making API calls.
    embeddings : list
        A list to store the generated embeddings.

    Methods
    -------
    embed_chunks(text_chunks, batch_size=100)
        Embed a list of text chunks and return the embeddings.
    get_embeddings()
        Retrieve the stored embeddings.
    save_embeddings(file_path)
        Save the embeddings to a file.
    load_embeddings(file_path)
        Load embeddings from a file.
    """
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.embeddings = []

    @diskcache_cache(cache_dir="embedding_cache", cache_size=10**9)  # 1GB
    def embed_chunks(self, text_chunks, batch_size=100):
        """
        Embed a list of text chunks and return the embeddings as a list of numpy arrays.

        Parameters
        ----------
        text_chunks : list
            List of text chunks to embed.
        batch_size : int, optional
            Number of chunks to process in each batch (default is 100).

        Returns
        -------
        list
            List of numpy arrays, each representing an embedding corresponding to an input text chunk.
        """
        all_embeddings = []

        for i in tqdm(range(0, len(text_chunks), batch_size), desc="Embedding chunks"):
            batch = text_chunks[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch, model="text-embedding-3-large"
            )

            batch_embeddings = [
                np.array(embedding.embedding) for embedding in response.data
            ]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def get_embeddings(self):
        """
        Get the stored embeddings.

        Returns:
        list: List of stored embeddings.
        """
        return self.embeddings

    def save_embeddings(self, file_path):
        """
        Save the embeddings to a file.

        Args:
        file_path (str): Path to save the embeddings file.
        """
        np.save(file_path, np.array(self.embeddings))

    def load_embeddings(self, file_path):
        """
        Load embeddings from a file.

        Args:
        file_path (str): Path to the embeddings file.

        Returns:
        list: List of loaded embeddings.
        """
        self.embeddings = np.load(file_path).tolist()
        return self.embeddings


if __name__ == "__main__":
    # Example usage
    embedder = Embedder()

    # Example text chunks
    text_chunks = [
        "Morgen ist das Wetter schön, also gehen wir golfen",
        "Wir spielen Golf auf dem Golfplatz in der Nähe des Hotels",
        "Am Abend essen wir im Restaurant des Hotels",
        "Das Hotel hat ein schönes Spa, wir gehen dort entspannen",
        "Am nächsten Tag gehen wir wandern in den Bergen",
        "Wir übernachten im Hotel und genießen die Ruhe",
        "Am Abend essen wir wieder im Hoteleigenen Restaurant",
    ]
    embedding = embedder.embed_chunks(text_chunks)

    print(f"{len(embedding[0])=}")
    assert len(embedding) == len(text_chunks)

    for i in range(len(embedding)):
        for j in range(i + 1, len(embedding)):
            print(f"{i=} {j=} {cosine_similarity([embedding[i]], [embedding[j]])=}")
