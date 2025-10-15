from dataclasses import dataclass

import gradio as gr

from ui.placeholders import INITIAL_CHATBOT_VALUE, QUESTION_PLACEHOLDER


@dataclass
class ChatUI:
    """
    A dataclass containing the chatbot, chat_input, and clear_btn.

    Attributes:
        chatbot (gr.Chatbot): The chatbot component.
        chat_input (gr.Textbox): The chat input component.
        clear_btn (gr.ClearButton): The clear button component.
    """

    chatbot: gr.Chatbot
    chat_input: gr.Textbox
    clear_btn: gr.ClearButton


def create_chat_ui() -> ChatUI:
    """
    Builds the left column of the UI, containing the chat interface.

    Returns:
        ChatUI: A dataclass containing the chatbot, chat_input, and clear_btn.
    """
    with gr.Column(scale=1):
        chatbot = gr.Chatbot(
            label="Chat History",
            type="messages",
            height=780,
            value=INITIAL_CHATBOT_VALUE,
        )
        chat_input = gr.Textbox(
            label="Your Question",
            placeholder=QUESTION_PLACEHOLDER,
            interactive=True,
            submit_btn=True,
        )
        clear_btn = gr.ClearButton()

    return ChatUI(chatbot, chat_input, clear_btn)
