# One image for every environment: the same container runs on a laptop and in Azure.
FROM python:3.12-slim

# uv (copied from its official image) installs exactly the versions pinned in uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Dependencies first: Docker caches this layer and rebuilds it only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the application code and the synthetic documents.
COPY app ./app
COPY data ./data

# Never run as root inside the container.
RUN useradd --create-home --uid 1000 appuser
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
