from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
import uvicorn
import argparse
import os
import logging
logger = logging.getLogger(__name__)

def prepare_agent_card(url: str) -> AgentCard:
    """Create the agent card for the tau2 purple agent."""
    skill = AgentSkill(
        id="block_building",
        name="Block building",
        description="Build block on the grid",
        tags=["blocks", "building"],
        examples=[],
    )
    return AgentCard(
        name="builder_agent",
        description="Builder agent for instructions related to building blocks",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


class DummyBuilderAgent():
 def __call__(self, *args, **kwargs):
     return "[BUILD];Green,0,50,0;Green,-200,50,0;Green,200,50,0;Green,0,50,0;Green,-200,50,0;Green,200,50,0;Green,-200,150,0;Green,-200,250,0;Green,0,150,0;Green,0,250,0;Green,200,150,0;Green,200,250,0;Purple,200,50,100;Purple,0,50,100;Purple,-200,50,100"

class BuilderAgentExecutor(AgentExecutor):
    """Executor for the builder purple agent."""

    def __init__(self, debug: bool = False):
        self.model = DummyBuilderAgent()
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.debug = debug

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        if self.debug:
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

        result = self.model()

        # Send response back via A2A
        await event_queue.enqueue_event(
            new_agent_text_message(result, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError

def main():
    parser = argparse.ArgumentParser(description="Run the builder agent (purple agent).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9018, help="Port to bind the server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--card-url", default="", help="URL for the agent card")
    args = parser.parse_args()

    debug_env = os.getenv("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug = args.debug or debug_env
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING)
    
    card_url = args.card_url
    if not card_url:
        if args.host == "0.0.0.0":
            card_host = "127.0.0.1"
        else:
            card_host = args.host
        card_url = f"http://{card_host}:{args.port}"
    
    logger.info(f"Starting builder agent on {args.host}:{args.port} with card URL: {card_url}")
    card = prepare_agent_card(card_url)

    request_handler = DefaultRequestHandler(
        agent_executor=BuilderAgentExecutor(debug=debug),
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
