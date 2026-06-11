FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into the system (no venv needed in Docker)
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Train the model at build time
RUN uv run python app/model/trainer.py

EXPOSE 7000

CMD uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-7000}
