from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    response: str
    documents: List[Document]
    answer: str
    max_retries: int
    web_search: str
    loop_step: int
