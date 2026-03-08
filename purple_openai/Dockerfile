FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent

# Copy project metadata first (layer-cache friendly)
COPY pyproject.toml README.md ./

# Copy all source modules (flat layout — server.py imports agent, state, etc. directly)
COPY agent.py prompts.py server.py state.py validators.py __init__.py ./

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --no-dev

# Run the A2A server from the project root so flat imports resolve correctly
ENTRYPOINT ["uv", "run", "python", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "9022"]
EXPOSE 9022
