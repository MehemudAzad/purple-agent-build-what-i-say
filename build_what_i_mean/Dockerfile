FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY pragmatic_builder pragmatic_builder
COPY data data

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["uv", "run", "pragmatic_builder/evaluator_proxy.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
