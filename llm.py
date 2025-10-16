import json
import logging
from dataclasses import dataclass
from typing import Generator, Optional

import httpx
from yarl import URL

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    vocab_type: int
    n_vocab: int
    n_ctx_train: int
    n_embd: int
    n_params: int
    size: int


@dataclass
class Model:
    model_id: str
    created_at: int
    owned_by: str
    metadata: ModelMetadata


class LLM:
    def __init__(self, base_url: URL):
        self.base_url = base_url
        self.models_url = base_url / "models"
        self.chat_url = base_url / "chat" / "completions"

    def get_models(self) -> list[Model]:
        """
        Get the available models for the LLM.

        Returns:
            list[Model]: The available models for the LLM.

        Raises:
            Exception: If an error occurs while fetching the models.
        """
        try:
            r = httpx.get(str(self.models_url), timeout=10)
            r.raise_for_status()
            data: list[dict] = r.json().get("data", [])

            logger.info(f"data: {data}")

            return [
                Model(
                    model_id=model.get("id"),
                    created_at=model.get("created"),
                    owned_by=model.get("owned_by"),
                    metadata=ModelMetadata(**model.get("meta")),
                )
                for model in data
                if self._is_valid_model(model)
            ]
        except Exception as e:
            logger.error(f"Failed to fetch models from {self.models_url}. Error: {e}")
            return []

    def _is_valid_model(self, model: dict) -> bool:
        """
        Check if the model is valid.

        Args:
            model (dict): The model to check.

        Returns:
            bool: True if the model is valid, False otherwise.
        """
        return (
            model.get("object")
            and model.get(
                "object",
            )
            == "model"
            and model.get("id")
            and model.get("created")
            and model.get("owned_by")
            and model.get("meta")
        )

    def chat(self, model: str, prompt: str) -> Optional[str]:
        """
        Chat with the LLM.

        Args:
            model: The model to use.
            prompt: The prompt to send to the LLM.

        Returns:
            Optional[str]: The response from the LLM if successful, None otherwise.
        """
        try:
            r = httpx.post(
                str(self.chat_url),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=300,
            )
            r.raise_for_status()
            return r.json().get("choices")[0].get("message").get("content")
        except Exception as e:
            logger.error(f"Failed to chat with {model}. Error: {e}")
            return None

    def chat_stream(self, model: str, prompt: str) -> Generator[str, None, None]:
        """
        Stream chat completions from the LLM. Yields incremental text chunks.

        Falls back to non-streaming if the server does not support streaming.

        Args:
            model: The model to use.
            prompt: The prompt to send to the LLM.

        Returns:
            Generator[str, None, None]: A generator that yields incremental text chunks.
        """
        try:
            with httpx.stream(
                "POST",
                str(self.chat_url),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
                timeout=300,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    # Expect lines like: b"data: {json}" or b"data: [DONE]"
                    if line.startswith("data: "):
                        data = line.removeprefix("data: ").strip()
                        if data == "[DONE]":
                            break
                        try:
                            payload = json.loads(data)
                            delta = (
                                payload.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content")
                            )
                            if delta:
                                yield delta
                        except Exception as parse_err:
                            logger.debug(
                                f"Failed to parse streaming chunk: {parse_err}"
                            )
                return
        except Exception as e:
            logger.error(
                f"Streaming failed for model {model}. Falling back to non-streaming. Error: {e}"
            )

        # Fallback to non-streaming
        full = self.chat(model, prompt)
        if not full:
            return
        yield full
