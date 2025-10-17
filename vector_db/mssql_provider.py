"""Microsoft SQL Server vector database provider implementation."""

import logging
import re
from typing import Optional

import pyodbc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_sqlserver import SQLServer_VectorStore
from yarl import URL

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class MSSQLProvider(DBProvider):
    """
    SQL Server-based vector DB provider using LangChain's SQLServer_VectorStore integration.

    Attributes:
        _ui_string (str): A string representation of the DB provider for the UI.
        db (SQLServer_VectorStore): LangChain-compatible vector store.

    Args:
        embeddings (HuggingFaceEmbeddings): HuggingFace embeddings instance.
        connection_string (str): Full ODBC connection string (including target DB).
        table (str): Table name to store vector embeddings.

    Example:
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> from vector_db.mssql_provider import MSSQLProvider
        >>> embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        >>> provider = MSSQLProvider(
        ...     embeddings=embeddings,
        ...     connection_string=(
        ...         "Driver={ODBC Driver 18 for SQL Server};"
        ...         "Server=localhost,1433;Database=docs;UID=sa;"
        ...         "PWD=StrongPassword!;TrustServerCertificate=yes;Encrypt=no;"
        ...     ),
        ...     table="embedded_docs",
        ... )
        >>> provider.get_relevant_documents(
        ...     query="What is OpenShift?",
        ...     search_type="similarity",
        ...     search_kwargs={"k": 10}
        ... )
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        connection_string: str,
        table: str,
    ) -> None:
        """
        Initialize the MSSQLProvider.

        Args:
            embeddings (HuggingFaceEmbeddings): HuggingFace-compatible embedding model instance.
            connection_string (str): Full ODBC connection string including target database name.
            table (str): Table name to store document embeddings.
        """
        super().__init__(embeddings)

        self._ensure_database_exists(connection_string)

        self.db = SQLServer_VectorStore(
            connection_string=connection_string,
            embedding_function=self.embeddings,
            table_name=table,
            embedding_length=self.embedding_length,
        )

        parse_url = URL(self._extract_server_address(connection_string))
        sanitized_url = f"{parse_url.host}:{parse_url.port or 1433}"
        self._ui_string = f"MSSQL at {sanitized_url}"

        logger.info(
            "Connected to MSSQL instance at %s (table: %s)",
            sanitized_url,
            table,
        )

    def _extract_server_address(self, connection_string: str) -> str:
        match = re.search(r"Server=([^;]+)", connection_string, re.IGNORECASE)
        return match.group(1) if match else "unknown"

    def _extract_database_name(self, connection_string: str) -> Optional[str]:
        match = re.search(r"Database=([^;]+)", connection_string, re.IGNORECASE)
        return match.group(1) if match else None

    def _build_connection_string_for_master(self, connection_string: str) -> str:
        parts = connection_string.split(";")
        updated_parts = [
            "Database=master" if p.strip().lower().startswith("database=") else p
            for p in parts
            if p
        ]
        return ";".join(updated_parts) + ";"

    def _ensure_database_exists(self, connection_string: str) -> None:
        database = self._extract_database_name(connection_string)
        if not database:
            raise RuntimeError("No database name found in connection string.")

        master_conn_str = self._build_connection_string_for_master(connection_string)
        try:
            with pyodbc.connect(master_conn_str, autocommit=True) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"IF DB_ID('{database}') IS NULL CREATE DATABASE [{database}]"
                )
                cursor.close()
        except pyodbc.ProgrammingError as e:
            if "1801" in str(e):
                logger.info("Database %s already exists, continuing", database)
                return
            logger.exception("Failed to ensure database '%s' exists", database)
            raise RuntimeError(
                f"Failed to ensure database '{database}' exists: {e}"
            ) from e
