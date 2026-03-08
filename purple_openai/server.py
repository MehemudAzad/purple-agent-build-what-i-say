"""
A2A server entry point for the purple agent.

Exposes an A2A-compliant endpoint that the green agent (benchmark) talks to.
Delegates all reasoning to ``agent.PurplePipeline`` (two-stage LLM pipeline).

Usage:
    python purple_openai/server.py --host 127.0.0.1 --port 9022 --debug
"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn
from dotenv import load_dotenv

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from agent import PurplePipeline

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent card
# ---------------------------------------------------------------------------

def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="block_building",
        name="Block building",
        description=(
            "Interprets natural-language instructions and places coloured "
            "blocks on a 3-D grid.  Uses a two-stage LLM pipeline with "
            "speaker-aware pragmatic inference."
        ),
        tags=["blocks", "building", "pragmatics"],
        examples=[],
    )
    return AgentCard(
        name="purple_openai_agent",
        description=(
            "LLM-powered purple agent (rita) for the Build-What-I-Mean benchmark. "
            "Two-stage pipeline: Stage 1 classifies ambiguity, Stage 2 generates "
            "[BUILD] coordinates.  Tracks per-speaker conventions across rounds."
        ),
        url=url,
        version="2.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


# ---------------------------------------------------------------------------
# A2A executor
# ---------------------------------------------------------------------------

class PurpleAgentExecutor(AgentExecutor):
    """Thin A2A wrapper around :class:`PurplePipeline`."""

    def __init__(self, pipeline: PurplePipeline, debug: bool = False) -> None:
        self._pipeline = pipeline
        self._debug = debug

    async def execute(
        self, context: RequestContext, event_queue: EventQueue,
    ) -> None:
        user_input = context.get_user_input()
        ctx_id = context.context_id or "default"

        if self._debug:
            logger.info("━" * 60)
            logger.info("[A2A] context_id=%s", ctx_id)
            logger.info("[A2A] user_input=%.300s", user_input)

        reply = await self._pipeline.handle_message(user_input, ctx_id)

        if self._debug:
            logger.info("[A2A] reply=%.300s", reply)

        await event_queue.enqueue_event(
            new_agent_text_message(reply, context_id=context.context_id),
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue,
    ) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the two-stage OpenAI purple agent.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=9022, help="Bind port")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--card-url", default="", help="Public agent-card URL")
    args = parser.parse_args()

    debug_env = os.getenv("AGENT_DEBUG", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    debug = args.debug or debug_env

    logging.basicConfig(
        level=logging.INFO if debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    card_url = args.card_url
    if not card_url:
        card_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        card_url = f"http://{card_host}:{args.port}"

    # Build the pipeline from environment variables
    pipeline = PurplePipeline.from_env(debug=debug)
    executor = PurpleAgentExecutor(pipeline=pipeline, debug=debug)
    card = prepare_agent_card(card_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    logger.info(
        "Starting purple agent on %s:%d  model=%s",
        args.host, args.port, os.environ.get("PURPLE_MODEL", "gpt-4o"),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
