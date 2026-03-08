from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class QAClient(Protocol):
    async def answer(self, *, question: str, target_structure: str) -> str:
        ...


class DummyQuestionAnswerer:
    async def answer(self, *, question: str, target_structure: str) -> str:
        return "Yellow"


@dataclass
class QuestionAnswerer:
    model: str
    api_key: str
    base_url: Optional[str] = None
    timeout: float = 30.0
    temperature: float = 0.2
    max_tokens: int = 256

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @classmethod
    def from_env(cls) -> Optional[QAClient]:
        mode = os.getenv("AGENT_QA_MODE", "openai").strip().lower()
        if mode == "dummy":
            return DummyQuestionAnswerer()
        if mode != "openai":
            return None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        model = "gpt-4o-mini"  # FIXED: Always use gpt-4o-mini for Q&A
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "256"))
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def answer(self, *, question: str, target_structure: str) -> str:
        system_prompt = (
            "You answer questions about a target block structure on a 3D grid.\n\n"
            "COORDINATE SYSTEM:\n"
            "- Blocks are specified as Color,x,y,z (e.g., Red,0,50,0)\n"
            "- Y-axis is vertical height: y=50 (ground), y=150 (1 block high), y=250 (2 blocks high), y=350 (3 blocks high), y=450 (4 blocks high)\n"
            "- X and Z are horizontal positions on the grid\n\n"
            "ANSWERING QUESTIONS:\n"
            "- For 'how many blocks' or 'how high' questions: Count the blocks and give the NUMBER (e.g., '4 blocks' or '3')\n"
            "- For color questions: List the colors (e.g., 'Red and Blue')\n"
            "- For position questions: Describe the location clearly (e.g., 'to the right' or 'behind')\n"
            "- Keep answers SHORT and DIRECT - just the essential information\n\n"
            "EXAMPLES:\n"
            "Q: How many blocks should be in the yellow stack?\n"
            "A: 4 blocks\n\n"
            "Q: How high should the red stack be?\n"
            "A: 3 blocks high\n\n"
            "Q: What color blocks are at position x=0?\n"
            "A: Green and Purple"
        )
        user_prompt = (
            f"Target structure:\n{target_structure}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely:"
        )

        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
            }
            
            # GPT-4o and newer models use max_completion_tokens instead of max_tokens
            if "gpt-4o" in self.model or "gpt-4-turbo" in self.model:
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                api_params["max_tokens"] = self.max_tokens
            
            response = await self._client.chat.completions.create(**api_params)
        except Exception as exc:
            logger.warning("OpenAI QA failed: %s", exc)
            return "Unable to answer the question right now."

        choice = response.choices[0].message
        content = (choice.content or "").strip()
        return content or "No answer."
