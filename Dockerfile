FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy lockfile first for better cache
COPY uv.lock ./

# Install uv and sync dependencies (including dev for tests)
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --dev

# Copy source and test directories
COPY pyproject.toml ./
COPY src ./src
COPY tests ./tests

# Copy data directories and fixtures
COPY data/ ./data/

# Copy repo hygiene files
COPY README.md LICENSE CITATION.cff .gitignore ./

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD uv run lcm doctor || exit 1

# Default: run verification
ENTRYPOINT ["uv", "run", "lcm"]
