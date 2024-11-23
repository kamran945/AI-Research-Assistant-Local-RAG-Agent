import os
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_groq.chat_models import ChatGroq
from langchain_ollama import ChatOllama

from dotenv import load_dotenv, find_dotenv

# Load the API keys from .env
load_dotenv(find_dotenv(), override=True)


from src.vectordb.create_vectordb import PinconeVectorDb


router_prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content="""
                    You are an expert at routing a user question to a vectorstore or web search.
                    The vectorstore contains documents related to "RAG (Retrieval Augmented Generation)" from Arxiv.
                    Carefully think about the user's query, then make a decision about whether the query should be routed to the vectorstore or websearch.
                    All questions related to RAG should be routed to the vectorstore, otherwise they should be routed to the websearch.
                    Respond in JSON format with only one key that is: 
                    "datasource": one of two options here "vectorstore" or "websearch"
                    """
        ),
        HumanMessage(content="{question}"),
    ],
)

router_instructions = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to "RAG (Retrieval Augmented Generation)" from Arxiv.
Carefully think about the user's query, then make a decision about whether the query should be routed to the vectorstore or websearch.
All questions related to RAG should be routed to the vectorstore, otherwise they should be routed to the websearch.
Respond in JSON format with only one key that is: 
"datasource": one of two options here "vectorstore" or "websearch"
"""

doc_grader_instructions = """
You are an expert grader, that grades the document, if it is relevant to a question or not.
If the the documents have keyword(s) or any semantic similarity to the question, then it is relevant. Otherwise irrelevant.
"""
doc_grader_prompt = """
Here is the document: {document}

Here is the question: {question}

Respond in JSON format with only one key:
"binary_answer": "yes" if relevant, "no" if not relevant.
"""

rag_instructions = """
You are a helpful question answering AI Assistant.
You have the following context:
context: {context}

This is the question:
question: {question}

Now think carefully about the question above, and answer the question appropriately using only the context provided.
Use three sentences maximum and keep the answer concise.

Answer:
"""

hallucination_grader_instructions = """
You are an expert Teacher grading the answers provided by the student.
You will be given FACTS and Student Answers.
Criteria to grade:
1. Ensure that the student question is grounded in facts.
2. Ensure that the student answer does not contain any "hallucinationed information" outside of facts.

Respond with "YES" if the answer is grounded in facts.
Respond with "NO" if the student answer contains any "hallucinationed information" outside of facts.

Also provide reasoning for the grade given by you.

"""

hallucination_grader_prompt = """
FACTS:
{documents}

STUDENT ANSWER: 
{answer}. 

Respond in JSON format with two keys:
"binary_answer": "yes" or "no".
"explanation": reasons for the grade given.
"""

answer_grader_instructions = """
Your a TEACHER tasked with grading answers provided by the STUDENT to a given question.

Question:
{question}

Student Answer:
{answer}

Criteria:
1. Ensure that the Student Answer helps to answer the question.

Respond with "yes" if the Student Answer meets the criteria. 
Any additional information provided by the STUDENT is also acceptable only if it helps in answering the question.
Respond with "no" if the Student Answer does not meet all the criteria.

Also provide reasoning for the grade given by you.

Provide result in JSON format with two keys:
"binary_answer": "yes" or "no"
"explanation": reasoning for the grade given by you
"""
answer_grader_prompt = """
QUESTION:
{question}
STUDENT ANSWER:
{generated_response}. 

Provide result in JSON format with two keys:
"binary_answer": "yes" or "no"
"explanation": reasoning for the grade given by you
"""
