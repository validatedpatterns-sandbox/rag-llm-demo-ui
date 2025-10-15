import logging
import re
from dataclasses import dataclass
from typing import Optional

import gradio as gr

from app_config import AppConfig
from llm import LLM
from ui.placeholders import INITIAL_PROMPT_VALUE, INITIAL_RETRIEVED_DOCS_VALUE
from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


@dataclass
class ConfigUI:
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


def get_model_choices(llms: list[LLM]) -> list[str]:
    """
    Get the model choices for the LLM.

    Args:
        llms (list[LLM]): The list of LLM providers.

    Returns:
        list[str]: The model choices.
    """
    if not llms:
        error_message = "No LLMs configured. Environment variable LLM_URLS must be set to a list of OpenAI API-compatible endpoints."
        logger.error(error_message)
        return [error_message]

    model_choices = [
        f"{model.model_id} ({llm.base_url})"
        for llm in llms
        for model in llm.get_models()
    ]

    if not model_choices:
        llm_urls = [str(llm.base_url) for llm in llms].join(", ")
        error_message = f"No models found for the LLM(s). Please check the LLM URL(s) and make sure they are OpenAI API-compatible endpoints. {llm_urls}"
        logger.error(error_message)
        return [error_message]

    return model_choices


def get_llm_model_from_choice(
    config: AppConfig, choice: str
) -> Optional[tuple[LLM, str]]:
    """
    Get the LLM and model from the choice.

    Args:
        config (AppConfig): The configuration object.
        choice (str): The choice of LLM.

    Returns:
        Optional[tuple[LLM, str]]: The LLM and model if found, None otherwise.
    """
    # Example choice: "model_id (http://localhost:8080)"
    match = re.match(r"^(.*)\s\((.*)\)$", choice)
    if not match:
        logger.error(
            f"Invalid choice: '{choice}'. Expected format: 'model_id (http://localhost:8080)'"
        )
        return None

    base_url = match.group(2)

    for llm in config.llms:
        if str(llm.base_url) == base_url:
            return llm, match.group(1)

    available_urls = ", ".join([str(llm.base_url) for llm in config.llms])
    logger.error(
        f"'{base_url}' not found in the configured LLM URLs: [{available_urls}]. Please check the LLM_URLS environment variable."
    )
    return None


def get_db_provider_choices(db_providers: list[DBProvider]) -> list[str]:
    """
    Get the DB provider choices.

    Args:
        db_providers (list[DBProvider]): The list of DB providers.

    Returns:
        list[str]: The DB provider choices.
    """
    if not db_providers:
        error_message = "No DB providers configured. Environment variable DB_PROVIDERS must be set to a list of database provider configurations."
        logger.error(error_message)
        return [error_message]

    return [db_provider.ui_string() for db_provider in db_providers]


def get_db_provider_from_choice(config: AppConfig, choice: str) -> Optional[DBProvider]:
    """
    Get the DB provider from the choice.

    Args:
        config (AppConfig): The configuration object.
        choice (str): The choice of DB provider.

    Returns:
        Optional[DBProvider]: The DB provider if found, None otherwise.
    """
    for db_provider in config.db_providers:
        if db_provider.ui_string() == choice:
            return db_provider

    available_choices = [
        db_provider.ui_string() for db_provider in config.db_providers
    ].join(", ")
    logger.error(
        f"'{choice}' not found in the configured DB provider choices: {available_choices}. Please check the DB_PROVIDERS environment variable."
    )
    return None


def create_config_ui(config: AppConfig) -> ConfigUI:
    """
    Builds the right column of the UI, with configuration and RAG internals.

    Args:
        config (AppConfig): The configuration object.

    Returns:
        ConfigUI: A dataclass containing the config UI components.
    """
    with gr.Column(scale=1):
        with gr.Accordion("Configuration", open=True):
            model_choices = get_model_choices(config.llms)
            llm_model_dd = gr.Dropdown(
                label="LLM Model",
                choices=model_choices,
                value=model_choices[0],
                interactive=True,
                info="Select the Large Language Model to use for generation.",
            )

            db_provider_choices = get_db_provider_choices(config.db_providers)
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
