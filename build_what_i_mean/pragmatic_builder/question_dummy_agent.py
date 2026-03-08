from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
import argparse
import logging
import os
import uvicorn

logger = logging.getLogger(__name__)


def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="block_building",
        name="Block building",
        description="Build block on the grid",
        tags=["blocks", "building"],
        examples=[],
    )
    return AgentCard(
        name="question_dummy_agent",
        description="Dummy agent that asks a question once, then returns a constant build.",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


class QuestionDummyAgent:
    def __init__(self):
        self._asked_by_context: dict[str, bool] = {}

    def respond(self, context_id: str | None) -> str:
        if context_id and not self._asked_by_context.get(context_id, False):
            self._asked_by_context[context_id] = True
            return "[ASK];which color is the block?"
        return (
            "[BUILD];Green,0,50,0;Green,-200,50,0;Green,200,50,0;"
            "Green,0,50,0;Green,-200,50,0;Green,200,50,0;"
            "Green,-200,150,0;Green,-200,250,0;Green,0,150,0;"
            "Green,0,250,0;Green,200,150,0;Green,200,250,0;"
            "Purple,200,50,100;Purple,0,50,100;Purple,-200,50,100"
        )


class QuestionDummyExecutor(AgentExecutor):
    def __init__(self, debug: bool = False):
        self._agent = QuestionDummyAgent()
        self._debug = debug

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        if self._debug:
            logger.info("-----")
            logger.info("Received request context: %s", context)
            logger.info("User input: %s", user_input)
            if context.message:
                parts = getattr(context.message, "parts", []) or []
                text_parts = []
                for part in parts:
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        text_parts.append(part.root.text)
                if text_parts:
                    logger.info("Raw message text: %s", "\n".join(text_parts))

        response = self._agent.respond(context.context_id)
        await event_queue.enqueue_event(
            new_agent_text_message(response, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Run the question dummy agent (purple agent).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9020, help="Port to bind the server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    debug_env = os.getenv("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug = args.debug or debug_env
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING)

    logger.info("Starting question dummy agent...")
    card = prepare_agent_card(f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=QuestionDummyExecutor(debug=debug),
        task_store=InMemoryTaskStore(),
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
