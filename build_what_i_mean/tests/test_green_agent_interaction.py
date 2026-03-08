import asyncio
import socket
import sys
import threading
from typing import Any, Dict
from pathlib import Path

import httpx
import pytest
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pragmatic_builder"))

from agentbeats.models import EvalRequest
from builder_agent import BuilderAgentExecutor, prepare_agent_card as prepare_builder_card
from question_dummy_agent import (
    QuestionDummyExecutor,
    prepare_agent_card as prepare_question_card,
)
from green_agent import BuildingInstructorGreenAgent
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore


class DummyUpdater:
    async def update_status(self, *args, **kwargs) -> None:
        return None

    async def add_artifact(self, *args, **kwargs) -> None:
        return None


class ServerThread(threading.Thread):
    def __init__(self, app, host: str, port: int):
        super().__init__(daemon=True)
        self._app = app
        self._host = host
        self._port = port
        self._server: uvicorn.Server | None = None

    def run(self) -> None:
        config = uvicorn.Config(self._app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        asyncio.run(self._server.serve())

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_for_agent(url: str, timeout: float = 5.0) -> None:
    async with httpx.AsyncClient(timeout=1) as client:
        end = asyncio.get_event_loop().time() + timeout
        while True:
            try:
                resp = await client.get(f"{url}/.well-known/agent-card.json")
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            if asyncio.get_event_loop().time() > end:
                raise RuntimeError(f"Agent at {url} did not become ready.")
            await asyncio.sleep(0.1)


def build_trials() -> Dict[str, Any]:
    return {
        "type": "building_game",
        "grid_context": "Test grid.",
        "chosen_list": 1,
        "first_speaker": "Lisa",
        "second_speaker": "Pia",
        "instructions_A": [
            {
                "round": 1,
                "speaker": "Lisa",
                "start_structure": "",
                "instruction": "Build a red block on the origin.",
                "trial_id": "t1",
                "list_id": 1,
                "target_structure": "Red,0,50,0",
            }
        ],
        "instructions_B": [],
    }


@pytest.mark.asyncio
async def test_green_agent_with_builder_agent(monkeypatch):
    trials = build_trials()
    monkeypatch.setattr("green_agent.BuildingGameTask.run", lambda *_: trials)

    port = get_free_port()
    host = "127.0.0.1"
    card = prepare_builder_card(f"http://{host}:{port}/")
    request_handler = DefaultRequestHandler(
        agent_executor=BuilderAgentExecutor(debug=False),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=request_handler).build()
    server = ServerThread(app, host, port)
    server.start()
    await wait_for_agent(f"http://{host}:{port}")

    agent = BuildingInstructorGreenAgent()
    req = EvalRequest(
        participants={"rita": f"http://{host}:{port}/"},
        config={
            "list1_path": "data/List1_FINAL_stimuli_list.csv",
            "list2_path": "data/List2_FINAL_stimuli_list.csv",
        },
    )
    result = await agent.run_eval(req, DummyUpdater())
    assert type(result.accuracy) is float
    assert type(result.avg_questions_per_instruction) is float

    server.stop()
    server.join(timeout=2)


@pytest.mark.asyncio
async def test_green_agent_with_question_dummy(monkeypatch):
    trials = build_trials()
    monkeypatch.setattr("green_agent.BuildingGameTask.run", lambda *_: trials)
    monkeypatch.setenv("AGENT_QA_MODE", "dummy")

    port = get_free_port()
    host = "127.0.0.1"
    card = prepare_question_card(f"http://{host}:{port}/")
    request_handler = DefaultRequestHandler(
        agent_executor=QuestionDummyExecutor(debug=False),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=request_handler).build()
    server = ServerThread(app, host, port)
    server.start()
    await wait_for_agent(f"http://{host}:{port}")

    agent = BuildingInstructorGreenAgent()
    req = EvalRequest(
        participants={"rita": f"http://{host}:{port}/"},
        config={
            "list1_path": "data/List1_FINAL_stimuli_list.csv",
            "list2_path": "data/List2_FINAL_stimuli_list.csv",
        },
    )
    result = await agent.run_eval(req, DummyUpdater())
    assert type(result.accuracy) is float
    assert type(result.avg_questions_per_instruction) is float

    server.stop()
    server.join(timeout=2)
