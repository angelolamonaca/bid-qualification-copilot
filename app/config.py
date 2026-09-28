"""Environment-specific configuration.

The same code runs on a laptop and in Azure; only these values change. Nothing here is secret:
authentication uses the Azure CLI login locally and the managed identity in Azure.

How it works: `Settings` lists every setting with a type and a default. pydantic-settings fills
each field from the environment variable with the same name (case-insensitive: CHAT_DEPLOYMENT
fills `chat_deployment`) or from the .env file, and checks the type. Think of it as a typed,
validated `process.env`.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Also read a local .env file. In Azure there is no .env: values come from real environment
    # variables set on the container. Unknown variables are ignored instead of causing errors.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Python note: `Literal[...]` allows only these exact values, like a TypeScript string union
    # ("local" | "azure" | "test"). A typo such as APP_ENV=prod fails at startup, not later.
    app_env: Literal["local", "azure", "test"] = "local"

    # --- AI models (Azure OpenAI in Microsoft Foundry) ---
    # Azure OpenAI v1 endpoint, e.g. https://<resource>.openai.azure.com/openai/v1/
    azure_openai_endpoint: str = ""
    chat_deployment: str = "gpt-5-mini"  # the deployment names chosen in Step 1
    embedding_deployment: str = "text-embedding-3-small"
    llm_timeout_seconds: float = 60.0  # give up on a single model call after this long
    llm_max_retries: int = 2  # retry rate limits and server errors at most this many times

    # --- Workflow state ---
    # "table" = Azure Table Storage (the Azurite emulator locally), "memory" = tests only
    state_backend: Literal["table", "memory"] = "table"
    tables_endpoint: str = ""  # Azure: https://<account>.table.core.windows.net (managed identity)
    tables_connection_string: str = ""  # local: Azurite emulator

    # Base URL of the (mock) CRM that approved bid registrations are written to
    crm_base_url: str = "http://localhost:8000/mock-crm"


# Python note: `@lru_cache` is a decorator (a wrapper around the function below). It remembers the
# first result and returns the same object on every later call, i.e. memoization. The environment
# is therefore read and validated only once per process.
@lru_cache
def get_settings() -> Settings:
    return Settings()
