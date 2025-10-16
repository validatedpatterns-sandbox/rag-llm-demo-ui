"""PostgreSQL with pgvector extension vector database provider implementation."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from yarl import URL

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class PGVectorProvider(DBProvider):
    """
    PostgreSQL with pgvector extension vector database provider.

    Attributes:
        db (PGVectorStore): LangChain-compatible PGVector client for vector storage.

    Args:
        embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
        url (str): PostgreSQL connection string (e.g., "postgresql://user:pass@host:5432/db").
        collection_name (str): Name of the table/collection used for storing vectors.

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.pgvector_provider import PGVectorProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        >>> provider = PGVectorProvider(
        ...     embeddings=embeddings,
        ...     url="postgresql://user:pass@localhost:5432/vector_db",
        ...     collection_name="rag_chunks"
        ... )
        >>> provider.get_relevant_documents(query="What is OpenShift?", search_type="similarity", search_kwargs={"k": 10})
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        url: str,
        collection_name: str,
    ):
        """
        Initialize a PGVectorProvider for use with PostgreSQL.

        Args:
            embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
            url (str): PostgreSQL connection string with pgvector enabled.
            collection_name (str): Name of the vector table in the database.
        """
        super().__init__(embeddings)

        engine = PGEngine.from_connection_string(url)
        engine.init_vectorstore_table(collection_name, self.embedding_length)

        self.db = PGVectorStore.create_sync(engine, self.embeddings, collection_name)

        parsed_url = URL(url)
        sanitized_url = f"{parsed_url.host}:{parsed_url.port or 5432}"
        self._ui_string = f"PGVector at {sanitized_url}"

        logger.info(
            "Connected to PGVector at %s (collection: %s)",
            sanitized_url,
            collection_name,
        )
