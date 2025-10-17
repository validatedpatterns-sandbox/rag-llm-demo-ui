"""Configuration management for RAG LLM demo application."""

import json
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from yarl import URL

from llm import LLM
from vector_db.db_provider import DBProvider
from vector_db.elastic_provider import ElasticProvider
from vector_db.mssql_provider import MSSQLProvider
from vector_db.pgvector_provider import PGVectorProvider
from vector_db.qdrant_provider import QdrantProvider
from vector_db.redis_provider import RedisProvider


@dataclass
class AppConfig:
    """
    Global configuration object for the application.

    Attributes:
        db_providers (list[DBProvider]): The list of database providers.
        llms (list[LLM]): The list of LLM providers.
    """

    db_providers: list[DBProvider]
    llms: list[LLM]

    @staticmethod
    def _get_required_env_var(key: str) -> str:
        """
        Retrieve a required environment variable or raise an error.

        Args:
            key (str): The environment variable name.

        Returns:
            str: The value of the environment variable.

        Raises:
            ValueError: If the variable is not defined.
        """
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} environment variable is required.")
        return value

    @staticmethod
    def _parse_log_level(log_level_name: str) -> int:
        """
        Convert a string log level into a `logging` module constant.

        Args:
            log_level_name (str): One of DEBUG, INFO, WARNING, ERROR, CRITICAL.

        Returns:
            int: Corresponding `logging` level.

        Raises:
            ValueError: If an invalid level is provided.
        """
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if log_level_name not in log_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: '{log_level_name}'. "
                f"Must be one of: {', '.join(log_levels.keys())}"
            )
        return log_levels[log_level_name]

    @staticmethod
    def _init_db_provider(
        provider: dict[str, str], logger: logging.Logger
    ) -> DBProvider:
        """
        Initialize a database provider from a dictionary of configuration.

        Args:
            provider (dict[str, str]): The configuration dictionary.

        Returns:
            DBProvider: The initialized database provider.

        Examples of provider configuration:
            ```json
            {
                "type": "qdrant",
                "collection": "docs",
                "url": "http://localhost:6333",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2"
            }
            ```

            ```json
            {
                "type": "redis",
                "index": "docs",
                "url": "redis://localhost:6379",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2"
            }
            ```

            ```json
            {
                "type": "elastic",
                "index": "docs",
                "url": "redis://localhost:6379",
                "password": "changeme",
                "user": "elastic",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2"
            }
            ```

            ```json
            {
                "type": "pgvector",
                "collection": "docs",
                "url": "postgresql+psycopg://user:pass@localhost:5432/mydb",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2"
            }
            ```

            ```json
            {
                "type": "mssql",
                "table": "docs",
                "connection_string": "Driver={ODBC Driver 18 for SQL Server}; \
Server=localhost,1433; \
Database=embeddings; \
UID=sa; \
PWD=StrongPassword!; \
TrustServerCertificate=yes; \
Encrypt=no;",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2"
            }
            ```
        """
        try:
            db_type = provider["type"].upper()
            embeddings = HuggingFaceEmbeddings(model_name=provider["embedding_model"])

            match db_type:
                case "QDRANT":
                    return QdrantProvider(
                        embeddings, provider["url"], provider["collection"]
                    )
                case "REDIS":
                    return RedisProvider(
                        embeddings,
                        provider["url"],
                        provider["index"],
                    )
                case "ELASTIC":
                    return ElasticProvider(
                        embeddings,
                        provider["url"],
                        provider["password"],
                        provider["index"],
                        provider["user"],
                    )
                case "PGVECTOR":
                    return PGVectorProvider(
                        embeddings,
                        provider["url"],
                        provider["collection"],
                    )
                case "MSSQL":
                    return MSSQLProvider(
                        embeddings,
                        provider["connection_string"],
                        provider["table"],
                    )
                case _:
                    raise ValueError(f"Invalid database type: {db_type}")

        except Exception:
            logger.error(
                f"Failed to initialize database provider: '{db_type}'. "
                "Check the DB_PROVIDERS environment variable."
            )
            raise

    @staticmethod
    def load() -> "AppConfig":
        """
        Load application settings from `.env` variables into a typed config object.

        Returns:
            Config: A fully-initialized configuration object.

        Raises:
            ValueError: If required environment variables are missing or malformed.
        """
        load_dotenv()
        get = AppConfig._get_required_env_var

        # Logging setup
        log_level = get("LOG_LEVEL").upper()
        logging.basicConfig(
            level=AppConfig._parse_log_level(log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
        logger.info("Logging initialized at level: %s", log_level)

        # Database providers
        try:
            db_providers = [
                AppConfig._init_db_provider(provider, logger)
                for provider in json.loads(get("DB_PROVIDERS"))
            ]
        except Exception:
            logger.error("Failed to initialize database providers.")
            raise

        # LLM providers
        try:
            llms = [LLM(URL(url)) for url in json.loads(get("LLM_URLS"))]
        except Exception:
            logger.error("Failed to initialize LLM providers.")
            raise

        return AppConfig(
            db_providers=db_providers,
            llms=llms,
        )
