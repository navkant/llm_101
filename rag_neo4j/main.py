import os
from typing import Optional, List, Tuple

from langchain.graphs import Neo4jGraph
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.chat_models import AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import SearchType
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field

from my_llm.my_lite_llm import MyLiteLLM

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")
raw_documents = WikipediaLoader(query="Elizabeth I", load_max_docs=2).load()

text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents[:3])
llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_API_BASE"),
    openai_api_version=os.environ.get("AZURE_API_VERSION"),
    openai_api_key=os.environ.get("AZURE_API_KEY"),
    deployment_name="gpt-4o-global",
)
llm_transformer = LLMGraphTransformer(llm=llm)


graph_documents = llm_transformer.convert_to_graph_documents(documents)


graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True,
)


vector = Neo4jVector(
    embedding=AzureOpenAIEmbeddings(
        azure_endpoint=os.environ.get("AZURE_API_BASE"),
        openai_api_version=os.environ.get("AZURE_API_VERSION"),
        openai_api_key=os.environ.get("AZURE_API_KEY"),
        model="text-embedding-3-small"
    ),
    url="bolt://localhost:7687",
    username="neo4j",
    password="12345678",
    database="neo4j",
).from_existing_graph(
    AzureOpenAIEmbeddings(
        azure_endpoint=os.environ.get("AZURE_API_BASE"),
        openai_api_version=os.environ.get("AZURE_API_VERSION"),
        openai_api_key=os.environ.get("AZURE_API_KEY"),
        model="text-embedding-3-small"
    ),
    search_type=SearchType.HYBRID,
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
    url="bolt://localhost:7687",
    username="neo4j",
    password="12345678",
    database="neo4j",
)


# extract entities from text
class Entities(BaseModel):
    """Identifying information about the entities"""

    names: List[str] = Field(
        ...,
        description="All the person, organization or business entities that appear in the text"
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text",
        ),
        (
            "human",
            "Use the given format to extract information from the following input question: {question}",
        ),
    ]
)
entity_chain = prompt | llm.with_structured_output(Entities)
print(entity_chain.invoke({"question": "Where was Amelia Earhart born?"}).names)











