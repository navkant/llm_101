from pathlib import Path
import sys

path = Path(__file__)
sys.path.append(str(path.resolve().parent.parent))
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from my_llm.my_lite_llm import MyLiteLLM


graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")
# graph.query(
#     """
#     MERGE (m:Movie {name:"Top Gun", runtime: 120})
#     WITH m
#     UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
#     MERGE (a:Actor {name:actor})
#     MERGE (a)-[:ACTED_IN]->(m)
#     """
# )

# graph.refresh_schema()
#
#
# print(graph.schema)


## ENHANCED SCHEMA
# enhanced_graph = Neo4jGraph(
#     url="bolt://localhost:7687",
#     username="neo4j",
#     password="12345678",
#     enhanced_schema=True,
# )
# print(enhanced_graph.schema)

my_lite_llm = MyLiteLLM(
    provider="azure",
    model="gpt-4o-global",
    temperature=0,
)

chain = GraphCypherQAChain.from_llm(
    my_lite_llm.get_chat_model(),
    graph=graph,
    verbose=True,
    top_k=2,
    allow_dangerous_requests=True,
    return_intermediate_steps=True,
)
result = chain.invoke({"query": "Who played in Top Gun?"})
print(f"Intermediate steps: {result['intermediate_steps']}")
print(f"Final answer: {result['result']}")
