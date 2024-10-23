from langchain_core.embeddings import Embeddings
from litellm import embedding


class LiteLLMEmbedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = embedding(
            model="azure/text-embedding-3-small",
            input=texts,
        )

        embedding_list = [
            data["embedding"] for data in response.data
        ]

        return embedding_list

    def embed_query(self, text: str) -> list[float]:
        response = embedding(
            model="azure/text-embedding-3-small",
            input=text,
        )

        return response.data[0]["embedding"]
