FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /workspace/llm-conceptual-modeling

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md LICENSE CITATION.cff .gitignore ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data

# Bake the validated GPU stack once so rented hosts do not spend paid time
# re-resolving torch, triton, and HF transfer settings.
RUN bash scripts/vast/bootstrap_gpu_host.sh /workspace/llm-conceptual-modeling

ENTRYPOINT ["uv", "run", "lcm"]
