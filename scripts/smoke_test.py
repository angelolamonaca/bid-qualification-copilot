"""Step 1 smoke test: call the chat model and the embedding model on Azure without any API key.

Run:
    uv run --python 3.12 --with openai --with azure-identity scripts/smoke_test.py <resource-name>
"""

import sys
import time

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

resource = sys.argv[1]

# DefaultAzureCredential tries several sign-in methods in order. On your laptop it finds your
# `az login` session; inside Azure Container Apps it will find the app's managed identity.
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

# The v1 endpoint accepts the standard OpenAI client. api_key receives a function that returns a
# fresh Entra ID token on every call, so no secret is ever stored anywhere.
client = OpenAI(
    base_url=f"https://{resource}.openai.azure.com/openai/v1/",
    api_key=token_provider,
)

start = time.perf_counter()
chat = client.chat.completions.create(
    model="gpt-5-mini",  # the *deployment* name we chose, not the model catalog name
    messages=[{"role": "user", "content": "In one sentence: what is a bid go/no-go decision?"}],
    reasoning_effort="low",
)
print(f"chat ({time.perf_counter() - start:.1f}s): {chat.choices[0].message.content}")
print(f"  tokens: {chat.usage.prompt_tokens} in / {chat.usage.completion_tokens} out")

start = time.perf_counter()
emb = client.embeddings.create(model="text-embedding-3-small", input=["bridge inspection services"])
vector = emb.data[0].embedding
elapsed = time.perf_counter() - start
print(f"embedding ({elapsed:.1f}s): {len(vector)} numbers, first three {vector[:3]}")
