"""
Two-stage LLM pipeline for the purple agent.

Flow per turn:
  1. Parse incoming message (instruction / feedback / answer / new-task / unknown)
  2. Route to the appropriate handler
  3. For instructions:
       a. Stage 1 (LLM) — classify ambiguity + extract known structure
       b. Decision gate  — resolve from speaker profile, or [ASK]
       c. Stage 2 (LLM) — generate [BUILD] string
  4. Validate output coordinates
  5. Update speaker profile from feedback
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Optional

from openai import AsyncOpenAI

from state import (
    MessageType,
    ParsedMessage,
    PendingClassification,
    SessionState,
    SpeakerProfile,
    StateManager,
    parse_message,
)
from prompts import (
    DIRECT_SYSTEM,
    STAGE1_SYSTEM,
    STAGE2_SYSTEM,
    stage1_user_prompt,
    stage2_user_prompt,
)
from validators import validate_build_response

logger = logging.getLogger(__name__)


class PurplePipeline:
    """Orchestrates the two-stage LLM pipeline with speaker-profile tracking."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        debug: bool = False,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._debug = debug
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )
        self._state_mgr = StateManager()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, debug: bool = False) -> PurplePipeline:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return cls(
            model=os.environ.get(
                "PURPLE_MODEL",
                os.environ.get("OPENAI_MODEL", "gpt-4o"),
            ).strip(),
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or None,
            temperature=float(os.environ.get("PURPLE_TEMPERATURE", "0.2")),
            max_tokens=int(os.environ.get("PURPLE_MAX_TOKENS", "1024")),
            timeout=float(os.environ.get("OPENAI_TIMEOUT", "60")),
            debug=debug,
        )

    # ==================================================================
    # Public entry point
    # ==================================================================

    async def handle_message(self, raw_text: str, context_id: str) -> str:
        """Receive a green-agent message, return a purple-agent response."""
        parsed = parse_message(raw_text)
        session = self._state_mgr.get_or_create(context_id)

        if self._debug:
            logger.info(
                "[purple] msg_type=%s  speaker=%s  context=%s",
                parsed.msg_type, parsed.speaker, context_id,
            )

        match parsed.msg_type:
            case MessageType.NEW_TASK:
                return await self._handle_new_task(session, parsed)
            case MessageType.FEEDBACK:
                return await self._handle_feedback(session, parsed)
            case MessageType.ANSWER:
                return await self._handle_answer(session, parsed)
            case MessageType.INSTRUCTION:
                return await self._handle_instruction(session, parsed)
            case _:
                return await self._handle_unknown(session, parsed)

    # ==================================================================
    # Message handlers
    # ==================================================================

    async def _handle_new_task(
        self, session: SessionState, parsed: ParsedMessage,
    ) -> str:
        """Reset state for a new game seed."""
        session.reset_for_new_seed()
        return "Acknowledged. Ready for the new task."

    # ----- feedback ----------------------------------------------------

    async def _handle_feedback(
        self, session: SessionState, parsed: ParsedMessage,
    ) -> str:
        """Parse feedback, update speaker profile, acknowledge."""
        self._update_profile_from_feedback(session, parsed)
        session.conversation.append({"role": "user", "content": parsed.raw})
        return "Acknowledged."

    # ----- answer to [ASK] --------------------------------------------

    async def _handle_answer(
        self, session: SessionState, parsed: ParsedMessage,
    ) -> str:
        """The green agent answered our clarifying question.  Build now."""
        session.conversation.append({"role": "user", "content": parsed.raw})

        # Extract the answer text (strip "Answer:" prefix and points notation)
        answer_text = parsed.raw
        if answer_text.startswith("Answer:"):
            answer_text = answer_text[7:].strip()
        paren_idx = answer_text.rfind("(")
        if paren_idx > 0:
            answer_text = answer_text[:paren_idx].strip()

        # Retrieve the last instruction from conversation history
        last_instruction, last_start = self._find_last_instruction(session)

        if last_instruction:
            resolution = f"Answer to clarification: {answer_text}"
            build = await self._stage2_build(
                last_instruction, last_start, resolution,
            )
        else:
            # Fallback: use direct prompt with recent history
            build = await self._direct_from_history(session)

        build = validate_build_response(build)
        session.conversation.append({"role": "assistant", "content": build})
        if session.pending:
            session.pending.our_build = build
        return build

    # ----- instruction (main two-stage pipeline) ----------------------

    async def _handle_instruction(
        self, session: SessionState, parsed: ParsedMessage,
    ) -> str:
        session.turn_count += 1
        speaker_name = parsed.speaker or "Unknown"
        speaker = session.get_or_create_speaker(speaker_name)
        speaker.turns_seen += 1
        session.current_speaker = speaker_name

        session.conversation.append({"role": "user", "content": parsed.raw})

        # ── Stage 1: Classify & Extract ──────────────────────────────
        classification = await self._stage1_classify(
            instruction=parsed.instruction_text or parsed.raw,
            start_structure=parsed.start_structure or "",
            speaker_name=speaker_name,
            speaker_summary=speaker.summary(),
        )

        if self._debug:
            logger.info("[stage1] %s", json.dumps(classification, indent=2))

        ambiguity = classification.get("ambiguity", "none")
        confidence = classification.get("confidence_in_build", 0.5)
        explicit_colors = classification.get("explicitly_mentioned_colors", [])
        explicit_counts = classification.get("explicitly_mentioned_counts", {})

        # ── Decision gate ────────────────────────────────────────────
        resolution: Optional[str] = None

        if ambiguity == "color":
            inferred = speaker.inferred_color()
            if inferred and len(speaker.color_fills) >= 1:
                resolution = (
                    f"The unspecified colour should be {inferred} "
                    f"(based on {speaker_name}'s established pattern: "
                    f"{speaker.color_fills})."
                )
                if self._debug:
                    logger.info("[gate] colour resolved from profile → %s", inferred)
            elif confidence < 0.6:
                return self._decide_ask(
                    session, speaker_name, ambiguity, classification, explicit_colors, explicit_counts,
                )

        elif ambiguity == "count":
            inferred = speaker.inferred_count()
            if inferred and len(speaker.count_fills) >= 1:
                resolution = (
                    f"The unspecified count should be {inferred} "
                    f"(based on {speaker_name}'s established pattern: "
                    f"{speaker.count_fills})."
                )
                if self._debug:
                    logger.info("[gate] count resolved from profile → %d", inferred)
            elif confidence < 0.6:
                return self._decide_ask(
                    session, speaker_name, ambiguity, classification, explicit_colors, explicit_counts,
                )

        # Store pending for feedback-driven profile update
        session.pending = PendingClassification(
            speaker_name=speaker_name,
            ambiguity_type=ambiguity,
            missing_description=classification.get("missing_description", ""),
            explicit_colors=explicit_colors,
            explicit_counts=explicit_counts,
        )

        # ── Stage 2: Generate [BUILD] ────────────────────────────────
        build = await self._stage2_build(
            instruction=parsed.instruction_text or parsed.raw,
            start_structure=parsed.start_structure or "",
            resolution=resolution,
        )

        build = validate_build_response(build)
        session.pending.our_build = build
        session.conversation.append({"role": "assistant", "content": build})
        return build

    # ----- unknown / retry --------------------------------------------

    async def _handle_unknown(
        self, session: SessionState, parsed: ParsedMessage,
    ) -> str:
        """Handle error / retry messages (e.g. 'Invalid response format')."""
        session.conversation.append({"role": "user", "content": parsed.raw})

        last_instruction, last_start = self._find_last_instruction(session)

        if last_instruction:
            resolution = (
                f"Your previous attempt had an error: {parsed.raw}  "
                "Please produce a correct [BUILD] response."
            )
            build = await self._stage2_build(
                last_instruction, last_start, resolution,
            )
        else:
            build = await self._direct_from_history(session)

        build = validate_build_response(build)
        session.conversation.append({"role": "assistant", "content": build})
        return build

    # ==================================================================
    # Helpers
    # ==================================================================

    def _decide_ask(
        self,
        session: SessionState,
        speaker_name: str,
        ambiguity: str,
        classification: dict,
        explicit_colors: list[str],
        explicit_counts: dict[str, int],
    ) -> str:
        """Return an [ASK] response and store pending classification."""
        question = classification.get(
            "suggested_question",
            "What colour should the unspecified blocks be?"
            if ambiguity == "color"
            else "How many of the unspecified blocks should there be?",
        )
        ask_response = f"[ASK];{question}"

        session.pending = PendingClassification(
            speaker_name=speaker_name,
            ambiguity_type=ambiguity,
            missing_description=classification.get("missing_description", ""),
            explicit_colors=explicit_colors,
            explicit_counts=explicit_counts,
        )
        session.conversation.append({"role": "assistant", "content": ask_response})

        if self._debug:
            logger.info("[gate] asking: %s", ask_response)
        return ask_response

    def _find_last_instruction(
        self, session: SessionState,
    ) -> tuple[Optional[str], str]:
        """Walk conversation backwards to find the most recent instruction."""
        for msg in reversed(session.conversation):
            if msg["role"] == "user" and "[TASK_DESCRIPTION]" in msg["content"]:
                pm = parse_message(msg["content"])
                return pm.instruction_text or pm.raw, pm.start_structure or ""
        return None, ""

    async def _direct_from_history(self, session: SessionState) -> str:
        """Produce a response using DIRECT_SYSTEM + recent conversation."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": DIRECT_SYSTEM},
        ]
        messages.extend(session.conversation[-8:])
        return await self._llm_call(messages)

    # ==================================================================
    # LLM calls
    # ==================================================================

    async def _stage1_classify(
        self,
        instruction: str,
        start_structure: str,
        speaker_name: str,
        speaker_summary: str,
    ) -> dict:
        """Stage 1: classify ambiguity and extract known facts."""
        user_msg = stage1_user_prompt(
            instruction, start_structure, speaker_name, speaker_summary,
        )
        messages = [
            {"role": "system", "content": STAGE1_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        raw = await self._llm_call(messages, temperature=0.1, max_tokens=512)

        try:
            clean = raw.strip()
            # Strip markdown code fences
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("[stage1] invalid JSON: %.200s", raw)
            return {
                "ambiguity": "none",
                "missing_description": "",
                "explicitly_mentioned_colors": [],
                "explicitly_mentioned_counts": {},
                "confidence_in_build": 0.7,
                "reasoning": "Stage 1 parse failure — proceeding with BUILD.",
            }

    async def _stage2_build(
        self,
        instruction: str,
        start_structure: str,
        resolution: str | None = None,
    ) -> str:
        """Stage 2: generate the [BUILD] response."""
        user_msg = stage2_user_prompt(instruction, start_structure, resolution)
        messages = [
            {"role": "system", "content": STAGE2_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        return await self._llm_call(messages)

    async def _llm_call(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Single OpenAI chat completions call with error handling."""
        try:
            params: dict = {
                "model": self._model,
                "messages": messages,
                "temperature": (
                    temperature if temperature is not None else self._temperature
                ),
            }
            mt = max_tokens if max_tokens is not None else self._max_tokens

            # Newer OpenAI models expect max_completion_tokens
            if any(
                tag in self._model
                for tag in ("gpt-4o", "gpt-4-turbo", "gpt-4.1", "o1", "o3", "o4")
            ):
                params["max_completion_tokens"] = mt
            else:
                params["max_tokens"] = mt

            completion = await self._client.chat.completions.create(**params)
            return (completion.choices[0].message.content or "").strip()

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return (
                "[ASK];I encountered an error processing this instruction. "
                "Could you repeat it?"
            )

    # ==================================================================
    # Profile update from feedback
    # ==================================================================

    def _update_profile_from_feedback(
        self, session: SessionState, parsed: ParsedMessage,
    ) -> None:
        """After green-agent feedback, update the speaker profile."""
        pending = session.pending
        if not pending:
            return

        speaker = session.get_or_create_speaker(pending.speaker_name)

        if parsed.is_correct is True:
            speaker.correct_builds += 1
        elif parsed.is_correct is False:
            speaker.incorrect_builds += 1

        # Extract fill values if we had an ambiguity and target is available
        if (
            pending.ambiguity_type in ("color", "count")
            and parsed.target_structure
        ):
            self._extract_fill_from_target(pending, speaker, parsed.target_structure)

        session.pending = None

    @staticmethod
    def _extract_fill_from_target(
        pending: PendingClassification,
        speaker: SpeakerProfile,
        target_structure: str,
    ) -> None:
        """Compare the target structure against what was explicitly mentioned
        to determine the actual fill value and record it in the speaker profile."""
        blocks = [b.strip() for b in target_structure.split(";") if b.strip()]
        target_colors: list[str] = []
        for block in blocks:
            parts = block.split(",")
            if len(parts) == 4:
                target_colors.append(parts[0].strip().capitalize())

        if not target_colors:
            return

        if pending.ambiguity_type == "color":
            # The fill colour = target colours NOT in the explicitly mentioned set
            explicit_set = {c.capitalize() for c in pending.explicit_colors}
            fill_colors = {c for c in target_colors if c not in explicit_set}
            for fc in fill_colors:
                speaker.color_fills.append(fc)
                logger.info(
                    "[profile] %s color_fill += %s", speaker.name, fc,
                )

        elif pending.ambiguity_type == "count":
            # Count is underspecified for some colour — find colours in target
            # whose count was not explicitly stated.
            target_counts = Counter(target_colors)
            explicit_count_keys = {
                k.capitalize() for k in pending.explicit_counts
            }
            for color, count in target_counts.items():
                if color not in explicit_count_keys:
                    speaker.count_fills.append(count)
                    logger.info(
                        "[profile] %s count_fill += %d (%s)",
                        speaker.name, count, color,
                    )
