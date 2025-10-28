"""LLM implementation."""

import json
import logging
from dataclasses import dataclass
from typing import Generator, Optional

import httpx
from yarl import URL

logger = logging.getLogger(__name__)


@dataclass
class Model:
    """
    A dataclass containing the model information.

    Attributes:
        id (str): The model ID.
    """

    id: str


class LLM:
    """
    A class representing the LLM.

    Attributes:
        base_url (URL): The base URL of the LLM.
        models (list[Model]): The available models for the LLM.
    """

    def __init__(self, base_url: URL):
        """
        Initialize the LLM.

        Args:
            base_url (URL): The base URL of the LLM.
        """
        self.base_url = base_url
        self._models_url = base_url / "models"
        self._chat_url = base_url / "chat" / "completions"
        self.models = self._get_models()

    def _get_models(self) -> list[Model]:
        """
        Get the available models for the LLM.

        Returns:
            list[Model]: The available models for the LLM.

        Raises:
            Exception: If an error occurs while fetching the models.
        """
        try:
            response = httpx.get(str(self._models_url), timeout=10)
            logger.debug("GET %s returned %s.", str(self._models_url), response.json())
            response.raise_for_status()
            data: list[dict] = response.json().get("data", [])

            return [Model(model["id"]) for model in data]
        except Exception:
            logger.error("Failed to fetch models from %s.", str(self._models_url))
            raise

    def chat(self, model: str, prompt: str) -> Optional[str]:
        """
        Chat with the LLM.

        Args:
            model: The model to use.
            prompt: The prompt to send to the LLM.

        Returns:
            Optional[str]: The response from the LLM if successful, None otherwise.

        Raises:
            Exception: If an error occurs while chatting with the LLM.
        """
        try:
            response = httpx.post(
                str(self._chat_url),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=300,
            )
            response.raise_for_status()
            return response.json().get("choices")[0].get("message").get("content")
        except Exception:
            logger.error(
                "Failed to chat with LLM %s. Model: %s.",
                str(self.base_url),
                model,
            )
            raise

    def chat_stream(self, model: str, prompt: str) -> Generator[str, None, None]:
        """
        Stream chat completions from the LLM. Yields incremental text chunks.

        Falls back to non-streaming if the server does not support streaming.

        Args:
            model: The model to use.
            prompt: The prompt to send to the LLM.

        Returns:
            Generator[str, None, None]: A generator that yields incremental text chunks.

        Raises:
            Exception: If an error occurs while streaming chat completions and the
            fallback to non-streaming fails.
        """
        try:
            with httpx.stream(
                "POST",
                str(self._chat_url),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
                timeout=300,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
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
                        except Exception:  # pylint: disable=broad-exception-caught
                            logger.warning(
                                "Failed to parse streaming chunk. Data: %s.",
                                data,
                                exc_info=True,
                            )
                return
        except Exception:  # pylint: disable=broad-exception-caught
            logger.error(
                "Streaming failed for LLM %s. Model: %s."
                " Falling back to non-streaming.",
                str(self.base_url),
                model,
                exc_info=True,
            )

        chat_response = self.chat(model, prompt)
        if not chat_response:
            return
        yield chat_response
