from langchain_core.documents import Document


def format_docs(docs: Document) -> str:
    """
    Format a list of documents into a single string, separated by newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)
