import os
import time
from getpass import getpass
import uuid

from tqdm.auto import tqdm

import pandas as pd

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec, Index
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import ArxivLoader
from langchain_core.vectorstores import VectorStoreRetriever

from dotenv import load_dotenv, find_dotenv

# Load the API keys from .env
load_dotenv(find_dotenv(), override=True)


class PinconeVectorDb:
    def __init__(
        self, cloud: str = os.getenv("CLOUD"), region: str = os.getenv("REGION")
    ):
        """
        Initialize the PinconeVectorDb class.
        Args:
            cloud (str): The cloud provider for Pinecone. Default is 'aws'.
            region (str): The AWS region for Pinecone. Default is 'us-east-1'.
        """
        # Check if 'PINECONE_API_KEY' is set; prompt if not
        self.pc_api_key = os.getenv("PINECONE_API_KEY") or getpass("Pinecone API key: ")
        self.pc, self.spec = self.initialize_pinecone_client(cloud=cloud, region=region)

    def initialize_pinecone_client(self, cloud: str, region: str):
        """
        Initialize the Pinecone client and return it.
        Args:
            cloud (str): The cloud provider for Pinecone.
            region (str): The AWS region for Pinecone.
        Returns:
            Pinecone client and serverless specification objects.
        """

        # Initialize the Pinecone client
        pc = Pinecone(api_key=self.pc_api_key)
        # Define the serverless specification for Pinecone (AWS region 'us-east-1')
        spec = ServerlessSpec(cloud=cloud, region=region)

        return pc, spec

    def create_pinecone_index(
        self,
        index_name: str = os.getenv("INDEX_NAME"),
        EMBEDDING_DIMS: int = int(os.getenv("EMBEDDING_DIMS")),
        metric: str = os.getenv("METRIC"),
    ) -> Index:
        """
        Creates a Pinecone index with the given name and dimensions.
        Args:
            index_name (str): The name of the index. Default is INDEX_NAME.
            EMBEDDING_DIMS (int): The dimensionality of the embeddings. Default is EMBEDDING_DIMS.
            metric (str): The metric used for similarity. Default is 'cosine'.
        Returns:
            Pinecone index object.
        """

        # Check if the index exists; create it if it doesn't
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                index_name,
                dimension=EMBEDDING_DIMS,  # Embedding dimension
                metric=metric,
                spec=self.spec,  # Cloud provider and region specification
            )

            # Wait until the index is fully initialized
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        else:
            print(f"Index {index_name} already exists.")

        # Connect to the index
        self.index = self.pc.Index(index_name)

        # Add a short delay before checking the stats
        time.sleep(1)

        # View the index statistics
        print(f"Index Stats:\n{self.index.describe_index_stats()}")


def get_embedding(
    model_name: str = os.getenv("EMBEDDING_MODEL_NAME"),
) -> HuggingFaceEmbeddings:
    """
    Initialize and return the HuggingFace Embeddings model.
    Args:
        model_name (str): The name of the HuggingFace model to use for embeddings. Default is EMBEDDING_MODEL_NAME.
    Returns:
        HuggingFaceEmbeddings model object.
    """
    # Initialize the HuggingFace Embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def get_docs(query: str = os.getenv("QUERY")):
    """
    Get documents using the ArxivLoader class.
    Args:
        query (str): The query string for the ArxivLoader. Default is QUERY.
    Returns:
        List of Document objects.
    """
    loader = ArxivLoader(query, load_max_docs=os.getenv("MAX_RESULTS"))
    documents = loader.load()
    return documents


def get_doc_chunks(documents: Document) -> Document:
    """
    Split a document into chunks using the RecursiveCharacterTextSplitter class.
    Args:
        documents (Document): The document to split.
    Returns:
        List of Document objects representing the chunks.
    """
    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    docs = doc_splitter.split_documents(documents)
    return docs


def get_pinecone_retriever(
    index_name: str = os.getenv("INDEX_NAME"),
    embedding: HuggingFaceEmbeddings = get_embedding(),
    search_type: str = os.getenv("RETRIEVER_SEARCH_TYPE"),
    k: int = int(os.getenv("RETRIEVER_K")),
) -> VectorStoreRetriever:
    """
    Create and return a VectorStoreRetriever object.
    Args:
        index_name (str): The name of the Pinecone index. Default is INDEX_NAME.
        embedding (HuggingFaceEmbeddings): The HuggingFace Embeddings model to use for embeddings. Default is get_embedding().
        search_type (str): The type of search to use. Default is RETRIEVER_SEARCH_TYPE.
        k (int): The number of results to retrieve. Default is RETRIEVER_K.
    Returns:
        VectorStoreRetriever object.
    """

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embedding
    )
    return docsearch.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


def create_pinecone_vectorstore(
    documents: Document,
    embedding: HuggingFaceEmbeddings = get_embedding(),
    index_name: str = os.getenv("INDEX_NAME"),
):
    pc = PinconeVectorDb()
    pc.create_pinecone_index()

    vs = PineconeVectorStore.from_documents(
        documents=documents, embedding=embedding, index_name=index_name
    )


if __name__ == "__main__":

    documents = get_docs()
    docs = get_doc_chunks(documents)
    uuids = [str(uuid.uuid4()) for _ in range(len(docs))]

    create_pinecone_vectorstore(documents=docs)
