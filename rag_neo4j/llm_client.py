from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os


def get_llm_client():
    return AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_API_BASE"),
        openai_api_version=os.environ.get("AZURE_API_VERSION"),
        openai_api_key=os.environ.get("AZURE_API_KEY"),
        deployment_name="gpt-4o-global",
    )


def get_embedding_model():
    return AzureOpenAIEmbeddings(
            azure_endpoint=os.environ.get("AZURE_API_BASE"),
            openai_api_version=os.environ.get("AZURE_API_VERSION"),
            openai_api_key=os.environ.get("AZURE_API_KEY"),
            model="text-embedding-3-small"
        )
