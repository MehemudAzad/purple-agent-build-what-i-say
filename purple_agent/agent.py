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
from validators import validate_build_response, validate_block

# Logging is configured by server.py (stdout for containers, file for local dev).
# This module just gets a named logger.
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
        logger.info(
            "[INIT] PurplePipeline initialized | Model: %s | Temp: %.1f | Timeout: %.1f | Debug: %s",
            self._model, self._temperature, timeout, self._debug
        )

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

        logger.info("\n" + "="*80)
        logger.info(
            "[INPUT] msg_type=%s | speaker=%s | context=%s | text_len=%d",
            parsed.msg_type, parsed.speaker, context_id, len(raw_text)
        )
        logger.info("Raw Message (first 200 chars):\n%s", raw_text[:200])
        logger.info("="*80)

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
        logger.info("[NEW_TASK] Resetting session state for new task")
        session.reset_for_new_seed()
        logger.info("[NEW_TASK] Session reset complete")
        return "Acknowledged. Ready for the new task."

    # ----- feedback ----------------------------------------------------

    async def _handle_feedback(
        self, session: SessionState, parsed: ParsedMessage,
    ) -> str:
        """Parse feedback, update speaker profile, acknowledge."""
        logger.info("[FEEDBACK] Processing feedback | is_correct=%s | target_len=%d", 
                   parsed.is_correct, len(parsed.target_structure or ""))
        self._update_profile_from_feedback(session, parsed)
        session.conversation.append({"role": "user", "content": parsed.raw})
        logger.info("[FEEDBACK] Feedback processed, returning acknowledgment")
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

        # Retrieve the last question we asked to provide context to Stage 2
        last_question = "a clarifying question"
        for msg in reversed(session.conversation):
            if msg.get("role") == "assistant" and msg.get("content", "").startswith("[ASK];"):
                last_question = msg["content"][6:].strip()
                break

        if last_instruction:
            resolution = f"We asked: '{last_question}' -> Answer: '{answer_text}'"
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
            # We asked a question and got an answer. This is no longer an
            # implicit convention to learn, so we stop tracking it as ambiguous.
            session.pending.ambiguity_type = "none"
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

        logger.info("[INSTRUCTION] Turn %d | Speaker: %s | Instruction: %s", 
                   session.turn_count, speaker_name, (parsed.instruction_text or parsed.raw)[:100])

        session.conversation.append({"role": "user", "content": parsed.raw})

        # ── Stage 1: Classify & Extract ──────────────────────────────
        logger.info("[STAGE1] Starting classification...")
        classification = await self._stage1_classify(
            instruction=parsed.instruction_text or parsed.raw,
            start_structure=parsed.start_structure or "",
            speaker_name=speaker_name,
            speaker_summary=speaker.summary(),
        )

        logger.info("[STAGE1] Classification result:\n%s", json.dumps(classification, indent=2))

        ambiguity = classification.get("ambiguity", "none")
        confidence = classification.get("confidence_in_build", 0.5)
        explicit_colors = classification.get("explicitly_mentioned_colors", [])
        explicit_counts = classification.get("explicitly_mentioned_counts", {})

        logger.info("[STAGE1] Ambiguity: %s | Confidence: %.2f | Colors: %s | Counts: %s",
                   ambiguity, confidence, explicit_colors, explicit_counts)

        # ── Decision gate ────────────────────────────────────────────
        logger.info("[GATE] Evaluating decision gate for ambiguity='%s'", ambiguity)
        resolution: Optional[str] = None

        # [VERSION 1.1] - Hardcoded ASK Failsafe
        if not explicit_colors and ambiguity != "none" and not speaker.inferred_color():
            logger.info("[GATE] Failsafe triggered: No explicit colors found and no profile inference")
            return self._decide_ask(session, speaker_name, "color", classification, explicit_colors, explicit_counts, parsed.start_structure or "")
            
        if not explicit_counts and ambiguity != "none" and not speaker.inferred_count():
            logger.info("[GATE] Failsafe triggered: No explicit counts found and no profile inference")
            return self._decide_ask(session, speaker_name, "count", classification, explicit_colors, explicit_counts, parsed.start_structure or "")

        if ambiguity == "color":
            inferred = speaker.inferred_color()
            if inferred and len(speaker.color_fills) >= 1:
                resolution = (
                    f"The unspecified colour should be {inferred} "
                    f"(based on {speaker_name}'s established pattern: "
                    f"{speaker.color_fills})."
                )
                logger.info("[GATE] Color resolved from profile → %s (pattern: %s)", inferred, speaker.color_fills)
            elif confidence < 0.6:
                logger.info("[GATE] Low confidence (%.2f < 0.6), asking for clarification on color", confidence)
                return self._decide_ask(
                    session, speaker_name, ambiguity, classification, explicit_colors, explicit_counts, parsed.start_structure or ""
                )
            else:
                logger.info("[GATE] No color pattern established yet, but confidence OK (%.2f)", confidence)

        elif ambiguity == "count":
            inferred = speaker.inferred_count()
            if inferred and len(speaker.count_fills) >= 1:
                resolution = (
                    f"The unspecified count should be {inferred} "
                    f"(based on {speaker_name}'s established pattern: "
                    f"{speaker.count_fills})."
                )
                logger.info("[GATE] Count resolved from profile → %d (pattern: %s)", inferred, speaker.count_fills)
            elif confidence < 0.6:
                logger.info("[GATE] Low confidence (%.2f < 0.6), asking for clarification on count", confidence)
                return self._decide_ask(
                    session, speaker_name, ambiguity, classification, explicit_colors, explicit_counts, parsed.start_structure or ""
                )
            else:
                logger.info("[GATE] No count pattern established yet, but confidence OK (%.2f)", confidence)

        # Store pending for feedback-driven profile update
        session.pending = PendingClassification(
            speaker_name=speaker_name,
            ambiguity_type=ambiguity,
            missing_description=classification.get("missing_description", ""),
            explicit_colors=explicit_colors,
            explicit_counts=explicit_counts,
            start_structure=parsed.start_structure or ""
        )

        # ── Stage 2: Generate [BUILD] ────────────────────────────────
        logger.info("[STAGE2] Starting coordinate generation...")
        if resolution:
            logger.info("[STAGE2] Using resolution: %s", resolution)
        build = await self._stage2_build(
            instruction=parsed.instruction_text or parsed.raw,
            start_structure=parsed.start_structure or "",
            resolution=resolution,
        )

        logger.info("[STAGE2] Generated build response: %s", build[:150])
        build = validate_build_response(build)
        logger.info("[STAGE2] Validated build: %s", build[:150])
        session.pending.our_build = build
        session.conversation.append({"role": "assistant", "content": build})
        return build

    # ----- unknown / retry --------------------------------------------

    async def _handle_unknown(
        self, session: SessionState, parsed: ParsedMessage,
    ) -> str:
        """Handle error / retry messages (e.g. 'Invalid response format')."""
        logger.info("[UNKNOWN] Handling unknown/error message: %s", parsed.raw[:100])
        session.conversation.append({"role": "user", "content": parsed.raw})

        last_instruction, last_start = self._find_last_instruction(session)

        if last_instruction:
            logger.info("[UNKNOWN] Found last instruction, retrying Stage 2 with error context")
            resolution = (
                f"Your previous attempt had an error: {parsed.raw}  "
                "Please produce a correct [BUILD] response."
            )
            build = await self._stage2_build(
                last_instruction, last_start, resolution,
            )
        else:
            logger.info("[UNKNOWN] No last instruction, using fallback history")
            build = await self._direct_from_history(session)

        build = validate_build_response(build)
        logger.info("[UNKNOWN] Final build: %s", build[:100])
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
        start_structure: str,
    ) -> str:
        """Return an [ASK] response and store pending classification."""
        # [VERSION 1.2] - Ask questions matching the question_answerer.py expected format.
        # The QA system (gpt-4o-mini) is prompted to answer:
        #   "How many [color] blocks should be in the stack?" → "4 blocks"
        #   "What color are the [description] blocks?"        → "Red and Blue"
        # Use the LLM's suggested_question from Stage 1 as the primary source.
        # Fall back to format-matched defaults only if Stage 1 gave nothing useful.
        llm_question = classification.get("suggested_question", "").strip()

        if llm_question:
            question = llm_question
        elif ambiguity == "color":
            missing = classification.get("missing_description", "the unspecified blocks")
            question = f"What color are {missing}?"
        else:
            # count ambiguity — ask in exactly the format the QA answerer expects
            colors = explicit_colors
            if colors:
                question = f"How many blocks should be in the {colors[0].lower()} stack?"
            else:
                question = "How many blocks should be in the target structure?"

        ask_response = f"[ASK];{question}"

        session.pending = PendingClassification(
            speaker_name=speaker_name,
            ambiguity_type=ambiguity,
            missing_description=classification.get("missing_description", ""),
            explicit_colors=explicit_colors,
            explicit_counts=explicit_counts,
            start_structure=start_structure,
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
        logger.info("[STAGE1_LLM] Calling LLM for classification...")
        logger.info("[STAGE1_LLM] User prompt (first 200 chars):\n%s", user_msg[:200])
        raw = await self._llm_call(messages, temperature=0.1, max_tokens=512)
        logger.info("[STAGE1_LLM] Raw response from LLM:\n%s", raw)

        try:
            clean = raw.strip()
            # Strip markdown code fences
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
            result = json.loads(clean)
            logger.info("[STAGE1_LLM] Successfully parsed JSON classification")
            return result
        except json.JSONDecodeError:
            logger.warning("[STAGE1_LLM] Invalid JSON: %.200s", raw)
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
        logger.info("[STAGE2_LLM] Calling LLM for coordinate generation...")
        logger.info("[STAGE2_LLM] User prompt (first 200 chars):\n%s", user_msg[:200])
        if resolution:
            logger.info("[STAGE2_LLM] Resolution provided: %s", resolution)
        
        # We need slightly higher tokens because we're doing Chain of Thought now.
        raw = await self._llm_call(messages, max_tokens=2048)
        logger.info("[STAGE2_LLM] Raw response from LLM:\n%s", raw)

        # Extract the actual [BUILD] string from the CoT response
        build_line = ""
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("[BUILD]"):
                build_line = line
                break
        
        if not build_line:
            logger.warning("[STAGE2_LLM] LLM failed to output a [BUILD] line, falling back to raw")
            build_line = raw

        # Auto-correct common LLM formatting mistakes like `Green(0,50,0)` instead of `Green,0,50,0;`
        if build_line.startswith("[BUILD]"):
            content = build_line[len("[BUILD]"):].strip()
            # If it forgot the semicolon after [BUILD]
            if content.startswith(";"):
                content = content[1:]
            
            # Replace parentheses if the LLM used them
            content = content.replace("(", ",").replace(")", ";")
            # Cleanup multiple semicolons caused by `);`
            content = content.replace(";;", ";")
            # Remove spaces
            content = content.replace(" ", "")
            # Remove trailing semicolons
            content = content.strip(";")
            
            build_line = f"[BUILD];{content}"

        return build_line

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
            logger.info("[PROFILE] No pending classification to update")
            return

        speaker = session.get_or_create_speaker(pending.speaker_name)

        if parsed.is_correct is True:
            speaker.correct_builds += 1
            logger.info("[PROFILE] Feedback: CORRECT | %s now has %d correct builds", 
                       pending.speaker_name, speaker.correct_builds)
        elif parsed.is_correct is False:
            speaker.incorrect_builds += 1
            logger.info("[PROFILE] Feedback: INCORRECT | %s now has %d incorrect builds", 
                       pending.speaker_name, speaker.incorrect_builds)

        # Extract fill values if we had an ambiguity and target is available
        if (
            pending.ambiguity_type in ("color", "count")
            and parsed.target_structure
        ):
            logger.info("[PROFILE] Learning from ambiguity='%s' with target: %s", 
                       pending.ambiguity_type, parsed.target_structure[:100])
            self._extract_fill_from_target(pending, speaker, parsed.target_structure)

        logger.info("[PROFILE] Updated speaker '%s': %s", pending.speaker_name, speaker.summary())
        session.pending = None

    @staticmethod
    def _extract_fill_from_target(
        pending: PendingClassification,
        speaker: SpeakerProfile,
        target_structure: str,
    ) -> None:
        """Compare the target structure against what was explicitly mentioned
        to determine the actual fill value and record it in the speaker profile."""
        logger.info("[EXTRACT] Extracting fill from target | speaker: %s | ambiguity: %s", 
                   speaker.name, pending.ambiguity_type)
        raw_target_blocks = [b.strip() for b in target_structure.split(";") if b.strip()]
        raw_start_blocks = [b.strip() for b in pending.start_structure.split(";") if b.strip()]

        # Normalize to ensure exact matching
        target_blocks = [validate_block(b) for b in raw_target_blocks if validate_block(b)]
        start_blocks = [validate_block(b) for b in raw_start_blocks if validate_block(b)]

        target_counter = Counter(target_blocks)
        start_counter = Counter(start_blocks)

        added_blocks = []
        for block, total_count in target_counter.items():
            diff = total_count - start_counter.get(block, 0)
            for _ in range(diff):
                added_blocks.append(block)

        logger.info("[EXTRACT] Added blocks (diff from start→target): %s", added_blocks)

        target_colors: list[str] = []
        for block in added_blocks:
            parts = block.split(",")
            if len(parts) == 4:
                target_colors.append(parts[0].strip().capitalize())

        if not target_colors:
            logger.info("[EXTRACT] No colors found in added blocks")
            return

        if pending.ambiguity_type == "color":
            # The fill colour = target colours NOT in the explicitly mentioned set
            explicit_set = {c.capitalize() for c in pending.explicit_colors}
            fill_colors = {c for c in target_colors if c not in explicit_set}
            logger.info("[EXTRACT] Explicit colors: %s | Fill colors: %s", explicit_set, fill_colors)
            for fc in fill_colors:
                speaker.color_fills.append(fc)
                logger.info(
                    "[EXTRACT] Learned: %s favors color %s", speaker.name, fc,
                )

        elif pending.ambiguity_type == "count":
            # Count is underspecified for some colour — find colours in target
            # whose count was not explicitly stated.
            target_counts = Counter(target_colors)
            explicit_count_keys = {
                k.capitalize() for k in pending.explicit_counts
            }
            logger.info("[EXTRACT] Target counts: %s | Explicit count keys: %s", target_counts, explicit_count_keys)
            for color, count in target_counts.items():
                if color not in explicit_count_keys:
                    speaker.count_fills.append(count)
                    logger.info(
                        "[profile] %s count_fill += %d (%s)",
                        speaker.name, count, color,
                    )
