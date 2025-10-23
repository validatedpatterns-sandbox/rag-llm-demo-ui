"""Configuration UI for the RAG LLM demo application."""

import logging
import re
from dataclasses import dataclass

import gradio as gr

from app_config import AppConfig
from llm import LLM
from ui.placeholders import INITIAL_PROMPT_VALUE, INITIAL_RETRIEVED_DOCS_VALUE
from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


@dataclass
class ConfigUI:  # pylint: disable=too-many-instance-attributes
    """
    A dataclass containing the config UI components.

    Attributes:
        llm_model_dd (gr.Dropdown): The dropdown for the LLM model.
        rag_provider_dd (gr.Dropdown): The dropdown for the RAG provider.
        search_type_radio (gr.Radio): The radio button for the search type.
        similarity_params_group (gr.Group): The group for the similarity parameters.
        similarity_k (gr.Number): The number input for the similarity k.
        threshold_params_group (gr.Group): The group for the threshold parameters.
        threshold_score (gr.Slider): The slider for the threshold score.
        mmr_params_group (gr.Group): The group for the mmr parameters.
        mmr_k (gr.Number): The number input for the mmr k.
        mmr_fetch_k (gr.Number): The number input for the mmr fetch k.
        mmr_lambda (gr.Slider): The slider for the mmr lambda.
        rag_docs_json (gr.JSON): The json for the retrieved documents.
        rag_final_prompt (gr.Code): The code for the final prompt.
    """

    llm_model_dd: gr.Dropdown
    rag_provider_dd: gr.Dropdown
    search_type_radio: gr.Radio
    similarity_params_group: gr.Group
    similarity_k: gr.Number
    threshold_params_group: gr.Group
    threshold_score: gr.Slider
    mmr_params_group: gr.Group
    mmr_k: gr.Number
    mmr_fetch_k: gr.Number
    mmr_lambda: gr.Slider
    rag_docs_json: gr.JSON
    rag_final_prompt: gr.Code


def get_model_choices(config: AppConfig) -> list[str]:
    """
    Get the model choices for the LLM.

    Args:
        config (AppConfig): The configuration object.

    Returns:
        list[str]: The model choices.

    Raises:
        ValueError: If no models choices are available.
    """
    try:
        model_choices = [
            f"{model.model_id} ({llm.base_url})"
            for llm in config.llms
            for model in llm.models
        ]

        if not model_choices:
            raise ValueError("No models choices available.")

        return model_choices
    except Exception:
        logger.error("Failed to get model choices.")
        raise


def get_llm_model_from_choice(config: AppConfig, choice: str) -> tuple[LLM, str]:
    """
    Get the LLM and model from the choice.

    Example of choice: "model_id (http://localhost:8080)"

    Args:
        config (AppConfig): The configuration object.
        choice (str): The choice of LLM.

    Returns:
        tuple[LLM, str]: The LLM and model.

    Raises:
        ValueError: If the choice is invalid or the LLM or model is not found.
    """
    try:
        match = re.match(r"^(.*)\s\((.*)\)$", choice)
        if not match:
            raise ValueError(
                f"Invalid choice: '{choice}'. Expected format: 'model_id (http://localhost:8080)'"
            )

        model_id = match.group(1)
        base_url = match.group(2)

        for llm in config.llms:
            if str(llm.base_url) == base_url:
                if model_id in [model.model_id for model in llm.models]:
                    return llm, model_id
                raise ValueError(
                    f"Model '{model_id}' not found for LLM: '{llm.base_url}'"
                )
        raise ValueError(f"LLM not found for base URL: '{base_url}'")

    except Exception:
        logger.error("Failed to get LLM model from choice '%s'.", choice)
        raise


def get_db_provider_choices(config: AppConfig) -> list[str]:
    """
    Get the DB provider choices.

    Args:
        config (AppConfig): The configuration object.

    Returns:
        list[str]: The DB provider choices.

    Raises:
        ValueError: If no DB provider choices are available.
    """
    try:
        choices = [db_provider.ui_string() for db_provider in config.db_providers]
        if not choices:
            raise ValueError("No DB provider choices available.")
        return choices

    except Exception:
        logger.error("Failed to get DB provider choices.")
        raise


def get_db_provider_from_choice(config: AppConfig, choice: str) -> DBProvider:
    """
    Get the DB provider from the choice.

    Args:
        config (AppConfig): The configuration object.
        choice (str): The choice of DB provider.

    Returns:
        DBProvider: The DB provider.

    Raises:
        ValueError: If the choice is invalid or the DB provider is not found.
    """
    try:
        for db_provider in config.db_providers:
            if db_provider.ui_string() == choice:
                return db_provider
        raise ValueError("Invalid choice.")
    except Exception:
        logger.error("Failed to get DB provider from choice '%s'.", choice)
        raise


def create_config_ui(config: AppConfig) -> ConfigUI:  # pylint: disable=too-many-locals
    """
    Builds the right column of the UI, with configuration and RAG internals.

    Args:
        config (AppConfig): The configuration object.

    Returns:
        ConfigUI: A dataclass containing the config UI components.
    """
    with gr.Column(scale=1):
        with gr.Accordion("Configuration", open=True):
            model_choices = get_model_choices(config)
            llm_model_dd = gr.Dropdown(
                label="LLM Model",
                choices=model_choices,
                value=model_choices[0],
                interactive=True,
                info="Select the Large Language Model to use for generation.",
            )

            db_provider_choices = get_db_provider_choices(config)
            rag_provider_dd = gr.Dropdown(
                label="RAG DB Provider",
                choices=db_provider_choices,
                value=db_provider_choices[0],
                interactive=True,
                info="Select the vector database for retrieval.",
            )

            gr.Markdown("### Retrieval Settings")
            search_type_radio = gr.Radio(
                ["similarity", "similarity_score_threshold", "mmr"],
                label="Search Type",
                value="similarity",
                info="Choose the retrieval strategy.",
            )

            with gr.Group(visible=True) as similarity_params_group:
                gr.Markdown("`similarity` arguments")
                similarity_k = gr.Number(
                    label="k",
                    value=4,
                    step=1,
                    precision=0,
                    info="The number of top documents to return.",
                    interactive=True,
                )

            with gr.Group(visible=False) as threshold_params_group:
                gr.Markdown("`similarity_score_threshold` arguments")
                threshold_score = gr.Slider(
                    label="score_threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    info="Return documents with a score above this threshold.",
                    interactive=True,
                )

            with gr.Group(visible=False) as mmr_params_group:
                gr.Markdown("`mmr` (Maximal Marginal Relevance) arguments")
                mmr_k = gr.Number(
                    label="k",
                    value=4,
                    step=1,
                    precision=0,
                    info="The number of documents to return.",
                    interactive=True,
                )
                mmr_fetch_k = gr.Number(
                    label="fetch_k",
                    value=20,
                    step=1,
                    precision=0,
                    info="Number of documents to fetch to pass to MMR algorithm.",
                    interactive=True,
                )
                mmr_lambda = gr.Slider(
                    label="lambda_mult",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    info="Balances diversity (1) vs. relevance (0).",
                    interactive=True,
                )

        gr.Markdown("## RAG Internals (Behind the Scenes)")

        with gr.Accordion("Step 1: Retrieved Documents", open=True):
            rag_docs_json = gr.JSON(
                label="Retrieved Documents (Full Data)",
                value=INITIAL_RETRIEVED_DOCS_VALUE,
            )

        with gr.Accordion("Step 2: Final Prompt to LLM", open=True):
            rag_final_prompt = gr.Code(
                label="Full Prompt Sent to LLM",
                language="markdown",
                value=INITIAL_PROMPT_VALUE,
            )

    return ConfigUI(
        llm_model_dd=llm_model_dd,
        rag_provider_dd=rag_provider_dd,
        search_type_radio=search_type_radio,
        similarity_params_group=similarity_params_group,
        similarity_k=similarity_k,
        threshold_params_group=threshold_params_group,
        threshold_score=threshold_score,
        mmr_params_group=mmr_params_group,
        mmr_k=mmr_k,
        mmr_fetch_k=mmr_fetch_k,
        mmr_lambda=mmr_lambda,
        rag_docs_json=rag_docs_json,
        rag_final_prompt=rag_final_prompt,
    )
