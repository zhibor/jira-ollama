# use whatever model you want to use
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate


def gen_response(context: str, query: str, model: str='llama3'):
    # Prompt
    prompt = PromptTemplate(
        template="""
        We have provided context information below. \n"
        "---------------------\n"
        "{context}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query}\n"
        )
    """,
        input_variables=["context", "query_str"],
    )

    # LLM
    llm = ChatOllama(model=model, temperature=0)

    # Chain
    chain = prompt | llm | StrOutputParser()

    # Run
    generation = chain.invoke({"context": context, "query": query})

    return generation


def simple_gen_response(query: str):
    # supports many more optional parameters. Hover on your `ChatOllama(...)`
    # class to view the latest available supported parameters
    llm = ChatOllama(model="llama3")
    prompt = ChatPromptTemplate.from_template("{query}")

    # using LangChain Expressive Language chain syntax
    # learn more about the LCEL on
    # /docs/expression_language/why
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"query": query})