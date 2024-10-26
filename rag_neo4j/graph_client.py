from langchain.graphs import Neo4jGraph


def get_graph_client():
    return Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")