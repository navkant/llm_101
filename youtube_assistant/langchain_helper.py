from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatLiteLLM
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from my_llm.lite_llm_embeddings import LiteLLMEmbedding


load_dotenv()


video_url = "https://www.youtube.com/watch?v=lG7Uxts9SXs"
def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    embeddings = LiteLLMEmbedding()

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query=query, k=k)
    docs_page_content = " ".join(d.page_content for d in docs)

    llm = ChatLiteLLM(
        model="azure/gpt-4o-global",
    )
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful youtube assistant that can answer questions about videos based on the video's transcript.
        Answer the following question: {question}
        By searching the following the video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        If you fee like you dont have enough information to answer the question, say "I don't know".
        Your answers should be detailed.
        """
    )

    chain = LLMChain(
        llm=llm, prompt=prompt
    )

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", " ")
    breakpoint()
    return response
