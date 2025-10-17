"""Abstract base class for vector database providers."""

import logging
from abc import ABC
from typing import Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
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
        _ui_string (str): A string representation of the DB provider for the UI.
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
        >>> provider.get_relevant_documents(query="What is OpenShift?",
        ...     search_type="similarity",
        ...     search_kwargs={"k": 10}
        ... )
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        """
        Initialize a DB provider with a HuggingFaceEmbeddings instance.

        Args:
            embeddings (HuggingFaceEmbeddings): The embeddings object used for vectorization.
        """
        self.embeddings: HuggingFaceEmbeddings = embeddings
        self.embedding_length: int = len(self.embeddings.embed_query("query"))
        self._ui_string: str = ""
        self.db: Optional[VectorStore] = None

    def ui_string(self) -> str:
        """
        Display a string representation of the DB provider for the UI.

        Returns:
            str: A string representation of the DB provider.

        Raises:
            Exception: If an error occurs while displaying the UI string.
        """
        try:
            if not self._ui_string:
                raise NotImplementedError(
                    f"UI string not implemented for DB provider of type {self.__class__.__name__}.",
                )
            return self._ui_string
        except Exception:
            logger.error(
                "Error displaying UI string for DB provider of type %s.",
                self.__class__.__name__,
            )
            raise

    def _as_retriever(
        self, search_type: str, search_kwargs: Optional[dict] = None
    ) -> VectorStoreRetriever:
        """
        Create a retriever from the vector database.

        Args:
            search_type (str): The type of search to perform.
            (one of "similarity", "similarity_score_threshold", "mmr")
            search_kwargs (Optional[dict]): The keyword arguments to pass to the search.

        Returns:
            VectorStoreRetriever: A retriever object.

        Raises:
            Exception: If an error occurs while creating the retriever.
        """
        try:
            if not self.db:
                raise NotImplementedError(
                    "Vector database not initialized for DB provider of type"
                    f" {self.__class__.__name__}.",
                )
            return self.db.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
        except Exception:
            logger.error(
                "Error creating retriever for DB provider of type %s.",
                self.__class__.__name__,
            )
            raise

    def get_relevant_documents(
        self,
        query: str,
        search_type: str = "similarity",
        search_kwargs: Optional[dict] = None,
    ) -> list[Document]:
        """
        Get the relevant documents from the vector database.

        Args:
            query (str): The query to search for.
            search_type (str): The type of search to perform.
            (one of "similarity", "similarity_score_threshold", "mmr")
            search_kwargs (Optional[dict]): The keyword arguments to pass to the search.

        Returns:
            list[Document]: A list of relevant documents.
        """
        try:
            return self._as_retriever(search_type, search_kwargs).invoke(query)
        except Exception:
            logger.error(
                "Error getting relevant documents for DB provider of type %s."
                " Query: %s, Search type: %s, Search kwargs: %s.",
                self.__class__.__name__,
                query,
                search_type,
                search_kwargs,
            )
            raise
