"""Main application entry point."""

from app_config import AppConfig
from ui.index import create_index_ui

config = AppConfig.load()

demo = create_index_ui(config)

if __name__ == "__main__":
    demo.launch(server_name=config.host, server_port=config.port)
