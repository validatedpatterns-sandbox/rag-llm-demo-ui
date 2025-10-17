"""Elasticsearch vector database provider implementation."""

import logging

from langchain_elasticsearch.vectorstores import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings
from yarl import URL

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class ElasticProvider(DBProvider):
    """
    Elasticsearch-backed vector DB provider using LangChain's ElasticsearchStore integration.

    Attributes:
        _ui_string (str): A string representation of the DB provider for the UI.
        db (ElasticsearchStore): LangChain-compatible Elasticsearch vector store.

    Args:
        embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
        url (str): Full URL to the Elasticsearch cluster (e.g., "http://localhost:9200").
        password (str): Password for the Elasticsearch user.
        index (str): Name of the Elasticsearch index to use.
        user (str): Elasticsearch username (e.g., "elastic").

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.elastic_provider import ElasticProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        >>> provider = ElasticProvider(
        ...     embeddings=embeddings,
        ...     url="http://localhost:9200",
        ...     password="changeme",
        ...     index="rag-docs",
        ...     user="elastic"
        ... )
        >>> provider.get_relevant_documents(
        ...     query="What is OpenShift?",
        ...     search_type="similarity",
        ...     search_kwargs={"k": 10}
        ... )
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        embeddings: HuggingFaceEmbeddings,
        url: str,
        password: str,
        index: str,
        user: str,
    ):
        """
        Initialize an Elasticsearch-based vector DB provider.

        Args:
            embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
            url (str): Full URL of the Elasticsearch service.
            password (str): Elasticsearch user's password.
            index (str): Name of the Elasticsearch index to use.
            user (str): Elasticsearch username (e.g., "elastic").
        """
        super().__init__(embeddings)

        self.db = ElasticsearchStore(
            embedding=self.embeddings,
            es_url=url,
            es_user=user,
            es_password=password,
            index_name=index,
        )

        parsed_url = URL(url)
        sanitized_url = f"{parsed_url.host}:{parsed_url.port or 9200}"
        self._ui_string = f"Elasticsearch at {sanitized_url}"

        logger.info(
            "Connected to Elasticsearch at %s (index: %s)", sanitized_url, index
        )
