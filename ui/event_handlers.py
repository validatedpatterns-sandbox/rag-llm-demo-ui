import logging
from typing import Optional

import gradio as gr
from langchain_core.documents import Document

from app_config import AppConfig
from ui.config import get_db_provider_from_choice, get_llm_model_from_choice
from ui.placeholders import (
    INITIAL_CHATBOT_VALUE,
    INITIAL_PROMPT_VALUE,
    INITIAL_RETRIEVED_DOCS_VALUE,
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
<s>[INST]
You are a helpful AI assistant. Your task is to answer the user's question taking into account the context documents and the chat history.

---
Context Documents:
{rag_docs}
---
Chat History:
{chatbot}
---
User's Question:
{chat_input}
[/INST]
"""


def update_search_kwargs_visibility(
    search_type: str,
) -> list[gr.Group, gr.Group, gr.Group]:
    """
    This function is triggered when the search type radio button changes.
    It returns a dictionary of updates for the visibility of the different
    search parameter groups.

    Args:
        search_type (str): The type of search to perform. (similarity, similarity_score_threshold, mmr)

    Returns:
        list[gr.Group, gr.Group, gr.Group]: A list of the visibility of the different search parameter groups.
    """
    return [
        gr.Group(visible=search_type == "similarity"),
        gr.Group(visible=search_type == "similarity_score_threshold"),
        gr.Group(visible=search_type == "mmr"),
    ]


def clear_chat_and_internals() -> list[gr.update]:
    """
    Resets the chat input, history, and RAG internal views to their
    initial placeholder states.

    Returns:
        list[gr.update]: A list of the updates to the different components.
    """
    return [
        gr.update(value=""),
        gr.update(value=INITIAL_CHATBOT_VALUE),
        gr.update(value=INITIAL_RETRIEVED_DOCS_VALUE),
        gr.update(value=INITIAL_PROMPT_VALUE),
    ]


def _retrieve_documents(
    config: AppConfig,
    chat_input: str,
    rag_provider_dd: str,
    search_type_radio: str,
    similarity_k: int,
    threshold_score: float,
    mmr_k: int,
    mmr_fetch_k: int,
    mmr_lambda: float,
) -> list[Document]:
    """
    Retrieves the relevant documents from the RAG provider.

    Args:
        config (AppConfig): The configuration object.
        chat_input (str): The chat input message.
        rag_provider_dd (str): The selected RAG provider.
        search_type_radio (str): The selected search type.
        similarity_k (int): The similarity k.
        threshold_score (float): The threshold score.
        mmr_k (int): The mmr k.
        mmr_fetch_k (int): The mmr fetch k.
        mmr_lambda (float): The mmr lambda.

    Returns:
        list[Document]: A list of relevant documents.
    """
    db_provider = get_db_provider_from_choice(config, rag_provider_dd)
    if not db_provider:
        logger.error(f"Invalid RAG provider: {rag_provider_dd}")
        return []

    search_kwargs = _get_search_kwargs(
        search_type_radio,
        similarity_k,
        threshold_score,
        mmr_k,
        mmr_fetch_k,
        mmr_lambda,
    )

    logger.debug(
        f"Searching '{db_provider.ui_string()}' with search type '{search_type_radio}' and search kwargs '{search_kwargs}'"
    )

    return db_provider.get_relevant_documents(
        chat_input, search_type_radio, search_kwargs
    )


def _get_search_kwargs(
    search_type_radio: str,
    similarity_k: int,
    threshold_score: float,
    mmr_k: int,
    mmr_fetch_k: int,
    mmr_lambda: float,
) -> dict:
    """
    Gets the keyword arguments for the search.

    Args:
        search_type_radio (str): The selected search type.
        similarity_k (int): The similarity k.
        threshold_score (float): The threshold score.
        mmr_k (int): The mmr k.
        mmr_fetch_k (int): The mmr fetch k.
        mmr_lambda (float): The mmr lambda.

    Returns:
        dict: A dictionary of the keyword arguments.
    """

    match search_type_radio:
        case "similarity":
            return {"k": similarity_k}
        case "similarity_score_threshold":
            return {"score_threshold": threshold_score}
        case "mmr":
            return {"k": mmr_k, "fetch_k": mmr_fetch_k, "lambda_mult": mmr_lambda}
        case _:
            logger.error(f"Invalid search type: {search_type_radio}")
            return {}


def _compile_final_prompt(
    chat_input: str, chatbot: list[dict], rag_docs: list[Document]
) -> str:
    """
    Compiles the final prompt to the LLM.

    Args:
        chat_input (str): The chat input message.
        chatbot (list[dict]): The chatbot history.
        rag_docs (list[Document]): A list of relevant documents.

    Returns:
        str: The final prompt.
    """

    return PROMPT_TEMPLATE.format(
        rag_docs=rag_docs,
        chatbot=chatbot,
        chat_input=chat_input,
    )


def _send_final_prompt_to_llm(
    config: AppConfig,
    llm_model_dd: str,
    final_prompt: str,
) -> Optional[str]:
    """
    Sends the final prompt to the LLM.

    Args:
        config (AppConfig): The configuration object.
        llm_model_dd: The selected LLM model.
        final_prompt (str): The final prompt.

    Returns:
        Optional[str]: The response from the LLM if successful, None otherwise.
    """
    llm_model = get_llm_model_from_choice(config, llm_model_dd)
    if not llm_model:
        logger.error(f"Invalid LLM model: {llm_model_dd}")
        return None
    llm, model = llm_model
    return llm.chat(model, final_prompt)


def submit_chat_message(
    config: AppConfig,
    chat_input: str,
    chatbot: list[dict],
    llm_model_dd: str,
    rag_provider_dd: str,
    search_type_radio: str,
    similarity_k: int,
    threshold_score: float,
    mmr_k: int,
    mmr_fetch_k: int,
    mmr_lambda: float,
) -> list[gr.update]:
    """
    Submits the user's chat message to the RAG pipeline.

    The RAG pipeline is as follows:
    1. Retrieve documents from the RAG provider.
    2. Compile the final prompt to the LLM.
    3. Send the final prompt to the LLM.
    4. Return the response from the LLM.

    Args:
        config (AppConfig): The configuration object.
        chat_input (str): The chat input message.
        chatbot (list[dict]): The chatbot history.
        llm_model_dd (str): The selected LLM model.
        rag_provider_dd (str): The selected RAG provider.
        search_type_radio (str): The selected search type.
        similarity_k (int): The similarity k.
        threshold_score (float): The threshold score.
        mmr_k (int): The mmr k.
        mmr_fetch_k (int): The mmr fetch k.
        mmr_lambda (float): The mmr lambda.

    Returns:
        list[gr.update]: A list of the updates to the different components.
    """

    # 1) Immediately reflect user message in chat history, clear input,
    #    and show busy state for retrieval and prompt preparation.
    user_first_history = chatbot + [{"role": "user", "content": chat_input}]
    yield [
        gr.update(value=user_first_history),
        gr.update(value=""),
        gr.update(value={"status": "Retrieving documents...", "query": chat_input}),
        gr.update(value="# Preparing final prompt..."),
    ]

    # 2) Retrieve documents and update immediately when available.
    rag_docs = _retrieve_documents(
        config,
        chat_input,
        rag_provider_dd,
        search_type_radio,
        similarity_k,
        threshold_score,
        mmr_k,
        mmr_fetch_k,
        mmr_lambda,
    )
    yield [
        gr.update(value=user_first_history),
        gr.update(value=""),
        gr.update(value=rag_docs),
        gr.update(value="# Preparing final prompt..."),
    ]

    # 3) Compile final prompt and update immediately.
    final_prompt = _compile_final_prompt(chat_input, chatbot, rag_docs)
    yield [
        gr.update(value=user_first_history),
        gr.update(value=""),
        gr.update(value=rag_docs),
        gr.update(value=final_prompt),
    ]

    # 4) Stream LLM response to the chat window.
    llm_model = get_llm_model_from_choice(config, llm_model_dd)
    if not llm_model:
        logger.error("Failed to resolve LLM model for streaming response.")
        error_history = user_first_history + [
            {
                "role": "assistant",
                "content": "An error occurred while getting the response from the LLM. Please try again.",
            }
        ]
        yield [
            gr.update(value=error_history),
            gr.update(value=""),
            gr.update(value=rag_docs),
            gr.update(value=final_prompt),
        ]
        return

    llm, model = llm_model
    assistant_history = user_first_history + [{"role": "assistant", "content": ""}]
    # Emit at least one empty assistant message so the UI shows a streaming bubble.
    yield [
        gr.update(value=assistant_history),
        gr.update(value=""),
        gr.update(value=rag_docs),
        gr.update(value=final_prompt),
    ]

    try:
        partial = ""
        for chunk in llm.chat_stream(model, final_prompt):
            partial += chunk
            assistant_history[-1]["content"] = partial
            yield [
                gr.update(value=assistant_history),
                gr.update(value=""),
                gr.update(value=rag_docs),
                gr.update(value=final_prompt),
            ]
    except Exception as e:
        logger.error(f"Error while streaming LLM response: {e}")
        assistant_history[-1][
            "content"
        ] = "An error occurred while streaming the response. Please try again."
        yield [
            gr.update(value=assistant_history),
            gr.update(value=""),
            gr.update(value=rag_docs),
            gr.update(value=final_prompt),
        ]
