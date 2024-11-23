from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from langchain_core.messages import SystemMessage, HumanMessage
import json
from langchain_core.documents import Document

from src.utils.state import GraphState
from src.vectordb.create_vectordb import get_pinecone_retriever
from src.utils.utils import format_docs
from src.utils.llm import llm, llm_json
from src.utils.prompts import (
    doc_grader_prompt,
    doc_grader_instructions,
    rag_instructions,
    router_instructions,
    hallucination_grader_instructions,
    hallucination_grader_prompt,
    answer_grader_instructions,
    answer_grader_prompt,
)


# Web Seearch
def web_search(state: GraphState) -> str:
    """
    Web search tool to find relevant documents using the Tavily API.
    Args:
        state (GraphState): The current state of the graph.
    Returns:
        str: The formatted results of the web search.
    """
    print("---- web_search ----")

    search = TavilyClient().search(
        query=state["question"], search_depth="advanced", max_results=5
    )
    results = search.get("results", [])

    formatted_results = "\n---\n".join(
        ["\n".join([x["title"], x["content"], x["url"]]) for x in results]
    )

    web_results = Document(page_content=formatted_results)
    # print(web_results)
    if "documents" in state.keys():
        state["documents"].append(web_results)

    return {"documents": [web_results]}


def retrieve(state: GraphState):
    print("---- retrieve ----")

    retriever = get_pinecone_retriever()
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


def grade_documents(state: GraphState):
    print("---- grade_documents ----")

    final_docs = []
    web_search = "no"
    for doc in state["documents"]:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=doc.page_content, question=state["question"]
        )

        result = llm_json.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        json_result = json.loads(result.content)

        if json_result["binary_answer"].lower() == "yes":
            print("---- relevant document ----")
            final_docs.append(doc)
        else:
            print("---- irrelevant document ----")
            web_search = "yes"
            continue

    return {"documents": final_docs, "web_search": web_search}


def generate_response(state: GraphState):
    print("---- generate_response ----")

    loop_step = state.get("loop_step", 0)

    rag_prompt = rag_instructions.format(
        context=format_docs(state["documents"]), question=state["question"]
    )

    result = llm.invoke([HumanMessage(content=rag_prompt)])

    return {"response": result.content, "loop_step": loop_step + 1}


def route_question(state: GraphState):
    print("---- route_question ----")

    result = llm_json.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    json_result = json.loads(result.content)
    print(f"---- route to: {json_result['datasource']} ----")

    if json_result["datasource"] == "vectorstore":
        return "vectorstore"
    else:
        return "websearch"


def decide_to_generate_response(state):
    print("---- decide_to_generate_response ----")

    if state["web_search"] == "yes":
        print("---- websearch: not all documents are relevant ----")
        return "websearch"
    else:
        print("---- generate_response: all documents are relevant ----")
        return "generate_response"


def check_hallucinations(state):
    print("---- check_hallucinations ----")

    max_retries = state.get("max_retries", 3)

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(state["documents"]), answer=state["response"]
    )

    result = llm_json.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )

    json_result = json.loads(result.content)

    if json_result["binary_answer"].lower() == "yes":
        print("---- no hallucination detected ----")

        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=state["question"], generated_response=state["response"]
        )

        grade_answer_result = llm_json.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )

        grade_answer_json_result = json.loads(grade_answer_result.content)

        if grade_answer_json_result["binary_answer"].lower() == "yes":
            print("---- answer is useful ----")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---- retrying generate_response ----")
            return "not_useful"
        else:
            print("---- max_retries reached ----")
            return "max_retries"

    elif state["loop_step"] <= max_retries:
        print("---- response is not grounded in documents, re-try ----")
        return "not_supported"
    else:
        print("---- max_retries reached ----")
        return "max_retries"
