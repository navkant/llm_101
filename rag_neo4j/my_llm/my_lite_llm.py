from langchain_core.embeddings import Embeddings
from litellm import embedding
from langchain_community.chat_models import ChatLiteLLM
from typing import List
from my_llm.lite_llm_embeddings import LiteLLMEmbedding


class MyLiteLLM:
    def __init__(self, provider: str = "azure", model: str = "gpt-4o-global", temperature: float = 0):
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def get_chat_model(self):
        return ChatLiteLLM(
            model=f"{self.provider}/{self.model}",
            temperature=self.temperature,
        )

    def get_embeddings(self, input: List[str]) -> List[List[float]]:
        response = embedding(
            model=f"{self.provider}/{self.model}",
            input=input,
        )

        embedding_list = [
            data["embedding"] for data in response.data
        ]
        return embedding_list

    def get_embedding(self, input: str) -> List[float]:
        response = embedding(
            model=f"{self.provider}/{self.model}",
            input=input,
        )

        return response.data[0]["embedding"]

    def get_embedding_model(self, embedding_model: str = "text-embedding-3-small") -> Embeddings:
        return LiteLLMEmbedding(provider=self.provider, model=embedding_model)
