"""Redis vector database provider implementation."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from yarl import URL

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class RedisProvider(DBProvider):
    """
    Redis-backed vector DB provider using RediSearch and LangChain's Redis integration.

    Attributes:
        db (RedisVectorStore): LangChain-compatible Redis vector store instance.

    Args:
        embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
        url (str): Redis connection string (e.g., "redis://localhost:6379").
        index (str): RediSearch index name to use for vector storage.

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.redis_provider import RedisProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        >>> provider = RedisProvider(
        ...     embeddings=embeddings,
        ...     url="redis://localhost:6379",
        ...     index="validated_docs"
        ... )
        >>> provider.get_relevant_documents(query="What is OpenShift?", search_type="similarity", search_kwargs={"k": 10})
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings, url: str, index: str):
        """
        Initialize a Redis-backed vector store provider.

        Args:
            embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
            url (str): Redis connection string.
            index (str): Name of the RediSearch index to use.
        """
        super().__init__(embeddings)

        metadata_schema = [{"name": "source", "type": "text"}]
        redis_cfg = RedisConfig.with_metadata_schema(
            metadata_schema, index_name=index, redis_url=url
        )

        self.db = RedisVectorStore(embeddings=self.embeddings, config=redis_cfg)

        parsed_url = URL(url)
        sanitized_url = f"{parsed_url.host}:{parsed_url.port or 6379}"
        self._ui_string = f"Redis at {sanitized_url}"

        logger.info("Connected to Redis at %s (index: %s)", sanitized_url, index)
