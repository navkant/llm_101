import os
from typing import Optional, List, Tuple


from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import SearchType
from langchain_core.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from graph_client import get_graph_client
from llm_client import get_llm_client, get_embedding_model


def save_to_neo4j():
    graph = get_graph_client()
    raw_documents = WikipediaLoader(query="Elizabeth I", load_max_docs=2).load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents[:3])
    llm = get_llm_client()
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True,
    )


def get_neo_vector():
    vector = Neo4jVector(
        embedding=get_embedding_model(),
        url="bolt://localhost:7687",
        username="neo4j",
        password="12345678",
        database="neo4j",
    ).from_existing_graph(
        embedding=get_embedding_model(),
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


def get_prompt():
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
    llm = get_llm_client()
    entity_chain = prompt | llm.with_structured_output(Entities)
    print(entity_chain.invoke({"question": "Where was Amelia Earhart born?"}).names)


def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]

    for word in words[:-1]:
        full_text_query += f"{word}~2 AND"

    full_text_query += f" {words[-1]}~2"

    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str) -> str:
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
    llm = get_llm_client()
    entity_chain = prompt | llm.with_structured_output(Entities)

    result = ""
    entities = entity_chain.invoke({"question": question})
    graph = get_graph_client()
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    for entity in entities.names:
        response = graph.query(
            """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node, score
                CALL {
                    WITH node
                    MATCH (node)-[r:!MENTIONS]->(neighbor)
                    RETURN  node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r:!MENTIONS]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)}
        )
        result += "\n".join([el['output'] for el in response])

    return result


if __name__ == "__main__":
    pass