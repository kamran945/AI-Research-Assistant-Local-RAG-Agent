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
from langchain_community.document_loaders import PubMedLoader
from langchain_core.vectorstores import VectorStoreRetriever

from dotenv import load_dotenv, find_dotenv

# Load the API keys from .env
load_dotenv(find_dotenv(), override=True)


def load_and_chunk_pdf(
    pdf_file_name: str,
    saved_dir: str = os.getenv("PDF_FOLDER"),
    chunk_size: int = int(os.getenv("CHUNK_SIZE")),
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP")),
) -> list[str]:
    """
    Loads a PDF file into chunks and returns a list of chunks.
    Args:
        pdf_file_name (str): The name of the PDF file.
        saved_dir (str): The directory where the PDF file is saved. Default is PDF_FOLDER.
        chunk_size (int): The size of each chunk in bytes. Default is CHUNK_SIZE.
        chunk_overlap (int): The overlap between chunks in bytes. Default is CHUNK_OVERLAP.
    Returns:
        List[str]: A list of chunks from the PDF file.
    """

    print(f"Loading and splitting into chunks: {pdf_file_name}")
    # name = remove_dot_from_filename(pdf_file_name)
    # print(name)

    pdf_file_path = os.path.join(saved_dir, pdf_file_name)

    # Load the PDF file into a DocumentLoader object
    loader = PyPDFLoader(pdf_file_path)
    data = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)

    return chunks


def add_chunks_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds chunks to the DataFrame, including their IDs and metadata.
    Args:
        df (pd.DataFrame): The DataFrame containing the paper details.
    Returns:
        pd.DataFrame: The updated DataFrame with added chunk information.
    """

    expanded_rows = []  # List to store expanded rows with chunk information

    # Loop through each row in the DataFrame
    for idx, row in df.iterrows():
        try:
            chunks = load_and_chunk_pdf(row["pdf_file_name"])
        except Exception as e:
            print(f"Error processing file {row['pdf_file_name']}: {e}")
            continue

        for i, chunk in enumerate(chunks):
            pre_chunk_id = i - 1 if i > 0 else ""  # Preceding chunk ID
            post_chunk_id = i + 1 if i < len(chunks) - 1 else ""  # Following chunk ID

            expanded_rows.append(
                {
                    "id": f"{row['arxiv_id']}#{i}",  # Unique chunk identifier
                    "title": row["title"],
                    "summary": row["summary"],
                    "authors": row["authors"],
                    "arxiv_id": row["arxiv_id"],
                    "url": row["url"],
                    "chunk": chunk.page_content,  # Text content of the chunk
                    "pre_chunk_id": (
                        "" if i == 0 else f"{row['arxiv_id']}#{pre_chunk_id}"
                    ),  # Previous chunk ID
                    "post_chunk_id": (
                        ""
                        if i == len(chunks) - 1
                        else f"{row['arxiv_id']}#{post_chunk_id}"
                    ),  # Next chunk ID
                }
            )
    # Return a new expanded DataFrame
    return pd.DataFrame(expanded_rows)


def get_embeddings(model_name, texts):
    # Define the Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    embeddings_list = []
    for text in texts:
        embeddings_list.append(embeddings.embed_query(text))

    return embeddings_list


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


class VectorStore:
    def __init__(self, index: Index):
        """
        Initialize the VectorStore class with the given Pinecone index.
        Args:
            index (Index): The Pinecone index to use for storing and searching vectors.
        """
        self.vectorstore = PineconeVectorStore(
            index=pc.index, embedding=self.get_embedding()
        )

    def get_embedding(
        self,
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
    Get documents using the PubMedLoader class.
    Args:
        query (str): The query string for the PubMedLoader. Default is QUERY.
    Returns:
        List of Document objects.
    """
    loader = PubMedLoader(query, load_max_docs=os.getenv("MAX_RESULTS"))
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
    vectorstore: PineconeVectorStore,
    search_type: str = os.getenv("RETRIEVER_SEARCH_TYPE"),
    k: int = int(os.getenv("RETRIEVER_K")),
) -> VectorStoreRetriever:
    """
    Initialize the VectorStoreRetriever class with the given search type and k.
    Args:
        vectorstore (PineconeVectorStore): The PineconeVectorStore to retrieve vectors from.
        search_type (str): The type of search to use. Default is RETRIEVER_SEARCH_TYPE.
        k (int): The number of results to return. Default is RETRIEVER_K.
    Returns:
        VectorStoreRetriever object.
    """
    # Initialize the VectorStoreRetriever class
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
    return retriever


if __name__ == "__main__":

    # Initialize the Pincone VectorDb class and create the index
    pc = PinconeVectorDb()
    pc.create_pinecone_index()

    # Initialize the VectorStore class with the created index
    vs = VectorStore(index=pc.index)
    vectorstore = vs.vectorstore

    documents = get_docs()
    docs = get_doc_chunks(documents)
    uuids = [str(uuid.uuid4()) for _ in range(len(docs))]

    vectorstore.add_documents(documents=docs, ids=uuids)
