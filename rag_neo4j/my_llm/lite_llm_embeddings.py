from langchain_core.embeddings import Embeddings
from litellm import embedding


class LiteLLMEmbedding(Embeddings):

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = embedding(
            model=f"{self.provider}/{self.model}",
            input=texts,
        )

        embedding_list = [
            data["embedding"] for data in response.data
        ]

        return embedding_list

    def embed_query(self, text: str) -> list[float]:
        response = embedding(
            model=f"{self.provider}/{self.model}",
            input=text,
        )

        return response.data[0]["embedding"]
