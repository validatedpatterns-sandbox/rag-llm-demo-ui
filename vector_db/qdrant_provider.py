"""Qdrant vector database provider implementation."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from yarl import URL

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class QdrantProvider(DBProvider):
    """
    Qdrant-backed vector database provider.

    Attributes:
        db (QdrantVectorStore): LangChain-compatible vector store.

    Args:
        embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
        url (str): Base URL for the Qdrant service (e.g., "http://localhost:6333").
        collection (str): Name of the Qdrant collection to use.

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.qdrant_provider import QdrantProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        >>> provider = QdrantProvider(
        ...     embeddings=embeddings,
        ...     url="http://localhost:6333",
        ...     collection="docs",
        ... )
        >>> provider.get_relevant_documents(query="What is OpenShift?", search_type="similarity", search_kwargs={"k": 10})
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        url: str,
        collection: str,
    ):
        """
        Initialize the Qdrant vector DB provider.

        Args:
            embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
            url (str): URL of the Qdrant instance.
            collection (str): Name of the collection to use or create.
        """
        super().__init__(embeddings)

        client = QdrantClient(
            url=url,
        )

        self._create_collection_if_not_exists(client, collection)

        self.db = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=self.embeddings,
        )

        parsed_url = URL(url)
        sanitized_url = f"{parsed_url.host}:{parsed_url.port or 6333}"
        self._ui_string = f"Qdrant at {sanitized_url}"

        logger.info(
            "Connected to Qdrant at %s (collection: %s)", sanitized_url, collection
        )

    def _create_collection_if_not_exists(
        self, client: QdrantClient, collection: str
    ) -> None:
        """
        Create a new collection in Qdrant using the computed embedding length.

        Args:
            client (QdrantClient): Qdrant client.
            collection (str): Name of the collection to create.
        """
        if not client.collection_exists(collection):
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=self.embedding_length, distance=Distance.COSINE
                ),
            )
