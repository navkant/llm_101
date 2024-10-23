from langchain_community.chat_models import ChatLiteLLM
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType

load_dotenv()


def generate_pet_name(animal_type: str, pet_color: str):
    llm = ChatLiteLLM(
        temperature=1,
        model="azure/gpt-4o-global",
    )
    prompt_template_name = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} Suggest me five cool "
                 "names for my pet."
    )
    name_chain = LLMChain(
        llm=llm, prompt=prompt_template_name, output_key="pet_name"
    )
    response = name_chain({"animal_type": animal_type, "pet_color": pet_color})

    return response


def langchain_agent():
    llm = ChatLiteLLM(
        temperature=1,
        model="azure/gpt-4o-global",
    )
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    result = agent.run(
        "what is average age of a dog. multiply it by 3"
    )
    print(result)


if __name__ == "__main__":
    # print(generate_pet_name("cow", "white"))
    langchain_agent()
