FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# ── Env ───────────────────────────────────────────────────────────────────────
# UV_PROJECT_ENVIRONMENT keeps the venv OUTSIDE the bind-mounted source dir,
# so `.:/workspace` at runtime does not shadow the pre-built virtualenv.
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── 1) Dependencies only (cached layer, project NOT installed yet) ────────────
# README.md is copied because pyproject.toml declares `readme = "README.md"`;
# hatchling validates metadata even for `--no-install-project`.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

# ── 2) Project source + editable install ─────────────────────────────────────
COPY . .
RUN uv sync --locked

# ── Default command ──────────────────────────────────────────────────────────
CMD ["python", "-m", "pytest", "-q", "tests/"]
