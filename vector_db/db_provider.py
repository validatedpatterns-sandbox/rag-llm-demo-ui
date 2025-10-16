"""Abstract base class for vector database providers."""

import logging
from abc import ABC

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class DBProvider(ABC):
    """
    Abstract base class for vector database providers.

    This class standardizes how vector databases are initialized and how documents
    are retrieved from them. All concrete implementations (e.g., Qdrant, Redis) must
    subclass `DBProvider`. Providers must initialize a vector store in the `__init__` method
    to `self.db` using the `embeddings` instance.

    Attributes:
        embeddings (HuggingFaceEmbeddings): An instance of HuggingFace embeddings.
        embedding_length (int): Dimensionality of the embedding vector.
        db (VectorStore): The vector store instance.

    Args:
        embeddings (HuggingFaceEmbeddings): A preconfigured HuggingFaceEmbeddings instance.

    Example:
        >>> class MyProvider(DBProvider):
        ...     def __init__(self, embeddings: HuggingFaceEmbeddings):
        ...         super().__init__(embeddings)
        ...         self.db = MyVectorStore(embeddings=embeddings)

        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        >>> provider = MyProvider(embeddings)
        >>> provider.get_relevant_documents(query="What is OpenShift?", search_type="similarity", search_kwargs={"k": 10})
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        """
        Initialize a DB provider with a HuggingFaceEmbeddings instance.

        Args:
            embeddings (HuggingFaceEmbeddings): The embeddings object used for vectorization.
        """
        self.embeddings: HuggingFaceEmbeddings = embeddings
        self.embedding_length: int = len(self.embeddings.embed_query("query"))

    def ui_string(self) -> str:
        """
        Display a string representation of the DB provider for the UI.

        Returns:
            str: A string representation of the DB provider.

        Raises:
            Exception: If an error occurs while displaying the UI string.
        """
        try:
            return self._ui_string
        except Exception as e:
            logger.error(f"Error displaying UI string: {e}")
            raise e

    def _as_retriever(
        self, search_type: str, search_kwargs: dict
    ) -> VectorStoreRetriever:
        """
        Create a retriever from the vector database.

        Args:
            search_type (str): The type of search to perform. (similarity, similarity_score_threshold, mmr)
            search_kwargs (dict): The keyword arguments to pass to the search.

        Returns:
            VectorStoreRetriever: A retriever object.

        Raises:
            Exception: If an error occurs while creating the retriever.
        """
        try:
            return self.db.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise e

    def get_relevant_documents(
        self, input: str, search_type: str = "similarity", search_kwargs: dict = {}
    ) -> list[Document]:
        """
        Get the relevant documents from the vector database.

        Args:
            input (str): The input to search for.
            search_type (str): The type of search to perform. (similarity, similarity_score_threshold, mmr)
            search_kwargs (dict): The keyword arguments to pass to the search.

        Returns:
            list[Document]: A list of relevant documents.
        """
        return self._as_retriever(search_type, search_kwargs).invoke(input)
