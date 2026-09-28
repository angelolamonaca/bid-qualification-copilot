"""Environment-specific configuration.

The same code runs on a laptop and in Azure; only these values change. Nothing here is secret:
authentication uses the Azure CLI login locally and the managed identity in Azure.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["local", "azure", "test"] = "local"

    # Azure OpenAI v1 endpoint, e.g. https://<resource>.openai.azure.com/openai/v1/
    azure_openai_endpoint: str = ""
    chat_deployment: str = "gpt-5-mini"
    embedding_deployment: str = "text-embedding-3-small"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # Workflow state: "table" = Azure Table Storage (Azurite emulator locally), "memory" = tests
    state_backend: Literal["table", "memory"] = "table"
    tables_endpoint: str = ""  # Azure: https://<account>.table.core.windows.net (managed identity)
    tables_connection_string: str = ""  # local: Azurite emulator

    # Base URL of the (mock) CRM that approved bid registrations are written to
    crm_base_url: str = "http://localhost:8000/mock-crm"


@lru_cache
def get_settings() -> Settings:
    return Settings()
