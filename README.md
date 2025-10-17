# RAG LLM Demo UI

A simple Gradio app that demonstrates a configurable Retrieval-Augmented Generation (RAG) flow. Pick an LLM, a vector database, tweak retrieval parameters, and inspect the RAG internals.

- [RAG LLM Demo UI](#rag-llm-demo-ui)
  - [Features](#features)
  - [How it Works](#how-it-works)
  - [Requirements](#requirements)
  - [Quickstart (Local)](#quickstart-local)
  - [Quickstart (Container)](#quickstart-container)
  - [Configuration](#configuration)
    - [Provider Examples](#provider-examples)
    - [Environment Loading](#environment-loading)
  - [Makefile Targets](#makefile-targets)
  - [License](#license)

## Features

- **Multiple Vector Databases**: Connect to Qdrant, Redis, Elasticsearch, pgvector, or SQL Server.
- **Any OpenAI-compatible LLM**: Provide one or more base URLs, and the UI will auto-discover available models.
- **Streaming Responses**: Get incremental updates from the assistant for a real-time chat experience.
- **Inspectable RAG Internals**: View the retrieved documents and the exact prompt sent to the LLM to understand the context.

## How it Works

The application follows a standard RAG pattern:

1.  **User Query**: The user asks a question in the chat interface.
2.  **Document Retrieval**: The app converts the query into an embedding and searches the selected vector database to find the most relevant document chunks.
3.  **Prompt Augmentation**: A prompt is constructed for the LLM, containing the original user query and the content of the retrieved documents as context.
4.  **LLM Generation**: The augmented prompt is sent to the selected LLM, which generates a response based on the provided context.
5.  **Stream to UI**: The response is streamed back to the user interface.

## Requirements

- **Conda**: Recommended for managing local Python environments.
- **Podman** or **Docker**: Required for running the application as a container.
- **ODBC Driver 18 for SQL Server**: Required _only_ if you are developing locally (not in a container) and need to connect to SQL Server. The container image includes this driver.

## Quickstart (Local)

1.  **Set up the Environment**:
    Create and activate the Conda environment.

    ```bash
    conda env create -f environment.yaml
    conda activate rag-llm-demo-ui
    ```

2.  **Configure your settings**:
    Copy the example environment file.

    ```bash
    cp .env.example .env
    ```

    Now you can edit .env with your LLM and DB provider URLs/credentials. This file is explicitly ignored in the [.gitignore](.gitignore). Even so, avoid adding any sensitive credentials to this file as a best practice.

    You can also export the variables directly in your shell:

    ```bash
    export LOG_LEVEL="INFO"
    export LLM_URLS='["http://localhost:1234/v1"]'
    export DB_PROVIDERS='[{"type": "qdrant", "collection": "docs", "url": "http://localhost:6333", "embedding_model": "sentence-transformers/all-mpnet-base-v2"}]'
    ```

3.  **Run the app**:
    ```bash
    make run
    ```
    The UI will be available at [`http://localhost:7860`](http://localhost:7860).

## Quickstart (Container)

1.  **Build the container image**:

    ```bash
    make build
    ```

2.  **Run the container**:
    The following command runs the container and maps it to your host's network, which is convenient for connecting to other services running on `localhost` (like an LLM or database).

    ```bash
    podman run --rm --network host \
      -e LOG_LEVEL="INFO" \
      -e LLM_URLS='["http://localhost:1234/v1"]' \
      -e DB_PROVIDERS='[{"type":"qdrant","collection":"docs","url":"http://localhost:6333","embedding_model":"sentence-transformers/all-mpnet-base-v2"}]' \
      localhost/rag-llm-demo-ui:latest
    ```

    The UI will be available at `http://localhost:7860`.

## Configuration

The application is configured via environment variables.

- `LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `LLM_URLS`: A JSON array of base URLs for [OpenAI-compatible APIs](https://github.com/openai/openai-openapi?tab=readme-ov-file). The app will query the `/models` and `/chat/completions` endpoints.
- `DB_PROVIDERS`: A JSON array where each object defines a connection to a vector database.

### Provider Examples

Below are examples for the `DB_PROVIDERS` variable.

**Qdrant**

```json
{
  "type": "qdrant",
  "collection": "docs",
  "url": "http://localhost:6333",
  "embedding_model": "sentence-transformers/all-mpnet-base-v2"
}
```

**Redis**

```json
{
  "type": "redis",
  "index": "docs",
  "url": "redis://localhost:6379",
  "embedding_model": "sentence-transformers/all-mpnet-base-v2"
}
```

**Elasticsearch**

```json
{
  "type": "elastic",
  "index": "docs",
  "url": "http://localhost:9200",
  "user": "elastic",
  "password": "changeme",
  "embedding_model": "sentence-transformers/all-mpnet-base-v2"
}
```

**pgvector (PostgreSQL)**

```json
{
  "type": "pgvector",
  "collection": "docs",
  "url": "postgresql+psycopg://user:pass@localhost:5432/mydb",
  "embedding_model": "sentence-transformers/all-mpnet-base-v2"
}
```

**SQL Server**

```json
{
  "type": "mssql",
  "table": "docs",
  "connection_string": "Driver={ODBC Driver 18 for SQL Server}; Server=localhost,1433; Database=embeddings; UID=sa; PWD=StrongPassword!; TrustServerCertificate=yes; Encrypt=no;",
  "embedding_model": "sentence-transformers/all-mpnet-base-v2"
}
```

### Environment Loading

- For local development, the app uses `python-dotenv` to load variables from a `.env` file.
- Variables set directly in the OS environment will always take precedence over those in the `.env` file.
- **SECURITY**: Do not commit secrets or credentials into `.env` files in your Git repository. Use `.gitignore` to exclude your local `.env` file. For production, inject these variables using your platform's secrets management tools.

## Makefile Targets

Common development tasks are available as `make` commands:

- `make install-deps`: Install pinned dependencies from `requirements.txt`.
- `make update-deps`: Re-resolve and update the `requirements.txt` lockfile (requires `pip-tools`).
- `make run`: Run the Gradio app locally with `python app.py`.
- `make build`: Build the container image using Podman.
- `make upload`: Push the container image to a registry.

## License

This project is licensed under the [Apcache V2.0 License](LICENSE).
