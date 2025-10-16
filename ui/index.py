import functools

import gradio as gr

from app_config import AppConfig
from ui.chat import create_chat_ui
from ui.config import create_config_ui
from ui.event_handlers import (
    clear_chat_and_internals,
    submit_chat_message,
    update_search_kwargs_visibility,
)


def create_index_ui(config: AppConfig) -> gr.Blocks:
    """
    Create the index UI.

    Args:
        config (AppConfig): The configuration object.

    Returns:
        gr.Blocks: The index UI.
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Configurable RAG Demo") as demo:
        gr.Markdown("# Configurable RAG Flow Demo")
        gr.Markdown(
            "Use the configuration panel to select models and tune retrieval parameters. The right panel shows the detailed output of the RAG process."
        )

        with gr.Row(equal_height=False):
            chat_ui = create_chat_ui()
            config_ui = create_config_ui(config)

        chat_ui.chat_input.submit(
            fn=functools.partial(submit_chat_message, config),
            inputs=[
                chat_ui.chat_input,
                chat_ui.chatbot,
                config_ui.llm_model_dd,
                config_ui.rag_provider_dd,
                config_ui.search_type_radio,
                config_ui.similarity_k,
                config_ui.threshold_score,
                config_ui.mmr_k,
                config_ui.mmr_fetch_k,
                config_ui.mmr_lambda,
            ],
            outputs=[
                chat_ui.chatbot,
                chat_ui.chat_input,
                config_ui.rag_docs_json,
                config_ui.rag_final_prompt,
            ],
        )

        chat_ui.clear_btn.click(
            fn=clear_chat_and_internals,
            inputs=None,
            outputs=[
                chat_ui.chat_input,
                chat_ui.chatbot,
                config_ui.rag_docs_json,
                config_ui.rag_final_prompt,
            ],
        )

        config_ui.search_type_radio.change(
            fn=update_search_kwargs_visibility,
            inputs=[config_ui.search_type_radio],
            outputs=[
                config_ui.similarity_params_group,
                config_ui.threshold_params_group,
                config_ui.mmr_params_group,
            ],
        )

    return demo
