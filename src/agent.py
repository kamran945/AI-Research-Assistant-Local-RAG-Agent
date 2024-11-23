from typing import Dict

from langgraph.graph import StateGraph, END, Graph
from IPython.display import Image, display

from dotenv import load_dotenv, find_dotenv

# Load the API keys from .env
load_dotenv(find_dotenv(), override=True)

from src.utils.state import GraphState

from src.utils.nodes import (
    web_search,
    retrieve,
    grade_documents,
    generate_response,
    route_question,
    decide_to_generate_response,
    check_hallucinations,
)


def get_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_response", generate_response)

    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("websearch", "generate_response")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_response,
        {
            "websearch": "websearch",
            "generate_response": "generate_response",
        },
    )
    workflow.add_conditional_edges(
        "generate_response",
        check_hallucinations,
        {
            "not_supported": "generate_response",
            "useful": END,
            "not_useful": "websearch",
            "max_retries": END,
        },
    )

    graph = workflow.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))

    return graph


def run_graph(graph: Graph, inputs: Dict) -> str:

    for event in graph.stream(inputs, stream_mode="values"):
        print(event)
    print(f"\nFINAL ANSWER: {event['response']}")

    return event["response"] if event["response"] else "No answer could be generated"


if __name__ == "__main__":

    inputs = {"question": "What is rag really?", "max_retries": 3}

    graph = get_graph()

    response = run_graph(graph=graph, inputs=inputs)
