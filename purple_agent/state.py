"""
Session state management for the purple agent.

Tracks:
- Per-speaker profiles (color/count fill conventions learned from feedback)
- Per-context conversation history
- Pending classification (carried between instruction → feedback)
- Message parsing utilities
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message classification patterns
# ---------------------------------------------------------------------------

_NEW_TASK_RE = re.compile(r"(?i)a new task is starting")
_FEEDBACK_RE = re.compile(r"^Feedback:")
_ANSWER_RE = re.compile(r"^Answer:")
_INSTRUCTION_RE = re.compile(r"\[TASK_DESCRIPTION\]")

# Extraction helpers
_SPEAKER_RE = re.compile(r"\[SPEAKER\]\s*(\S+)")
_START_STRUCT_RE = re.compile(r"\[START_STRUCTURE\]\s*(.*?)(?:\n|$)")

# Target structure from feedback
_CORRECT_TARGET_RE = re.compile(r"points\.\s*(.+?)\s*\|")
_INCORRECT_TARGET_RE = re.compile(r"Expected:\s*(.+),\s+but got:")


# ---------------------------------------------------------------------------
# Speaker profile
# ---------------------------------------------------------------------------

@dataclass
class SpeakerProfile:
    """Convention tracker for one speaker within a game session."""

    name: str
    color_fills: list[str] = field(default_factory=list)
    count_fills: list[int] = field(default_factory=list)
    turns_seen: int = 0
    correct_builds: int = 0
    incorrect_builds: int = 0

    # -- inference ----------------------------------------------------------

    def inferred_color(self) -> Optional[str]:
        """Most common color used for underspecified slots."""
        if not self.color_fills:
            return None
        return Counter(self.color_fills).most_common(1)[0][0]

    def inferred_count(self) -> Optional[int]:
        """Most common count used for underspecified slots."""
        if not self.count_fills:
            return None
        return Counter(self.count_fills).most_common(1)[0][0]

    # -- serialisation for prompt injection ---------------------------------

    def summary(self) -> str:
        lines = [
            f"Speaker '{self.name}' "
            f"({self.turns_seen} turns, {self.correct_builds} correct, "
            f"{self.incorrect_builds} incorrect):"
        ]
        if self.color_fills:
            lines.append(
                f"  When color was unspecified, they used: "
                f"{self.color_fills} → likely '{self.inferred_color()}'"
            )
        if self.count_fills:
            lines.append(
                f"  When count was unspecified, they used: "
                f"{self.count_fills} → likely {self.inferred_count()}"
            )
        if not self.color_fills and not self.count_fills:
            lines.append("  No underspecification patterns observed yet.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pending classification (bridges instruction → feedback)
# ---------------------------------------------------------------------------

@dataclass
class PendingClassification:
    """Stored between an instruction response and the subsequent feedback
    so that we can update the speaker profile with the actual fill values."""

    speaker_name: str
    ambiguity_type: str                                # "none" | "color" | "count"
    missing_description: str = ""
    explicit_colors: list[str] = field(default_factory=list)
    explicit_counts: dict[str, int] = field(default_factory=dict)
    our_build: Optional[str] = None
    start_structure: str = ""


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class MessageType:
    INSTRUCTION = "instruction"
    FEEDBACK    = "feedback"
    ANSWER      = "answer"
    NEW_TASK    = "new_task"
    UNKNOWN     = "unknown"


@dataclass
class ParsedMessage:
    """Result of classifying an incoming green-agent message."""

    msg_type: str
    raw: str
    speaker: Optional[str] = None
    start_structure: Optional[str] = None
    instruction_text: Optional[str] = None
    is_correct: Optional[bool] = None
    target_structure: Optional[str] = None


def parse_message(text: str) -> ParsedMessage:
    """Classify and parse an incoming message from the green agent."""
    text = text.strip()

    # -- new task ---------------------------------------------------------
    if _NEW_TASK_RE.search(text):
        return ParsedMessage(msg_type=MessageType.NEW_TASK, raw=text)

    # -- feedback ---------------------------------------------------------
    if _FEEDBACK_RE.match(text):
        is_correct = "Correct structure built!" in text
        target: Optional[str] = None
        if is_correct:
            m = _CORRECT_TARGET_RE.search(text)
            if m:
                target = m.group(1).strip()
        else:
            m = _INCORRECT_TARGET_RE.search(text)
            if m:
                target = m.group(1).strip()
        return ParsedMessage(
            msg_type=MessageType.FEEDBACK,
            raw=text,
            is_correct=is_correct,
            target_structure=target,
        )

    # -- answer to [ASK] --------------------------------------------------
    if _ANSWER_RE.match(text):
        return ParsedMessage(msg_type=MessageType.ANSWER, raw=text)

    # -- instruction ------------------------------------------------------
    if _INSTRUCTION_RE.search(text):
        speaker: Optional[str] = None
        m = _SPEAKER_RE.search(text)
        if m:
            speaker = m.group(1)

        start_structure = ""
        m = _START_STRUCT_RE.search(text)
        if m:
            start_structure = m.group(1).strip()

        # Instruction text = everything after the last tag-prefixed line
        lines = text.split("\n")
        instruction_lines: list[str] = []
        past_tags = False
        for line in lines:
            if past_tags:
                instruction_lines.append(line)
            elif not line.startswith("["):
                past_tags = True
                instruction_lines.append(line)
        instruction_text = "\n".join(instruction_lines).strip()

        return ParsedMessage(
            msg_type=MessageType.INSTRUCTION,
            raw=text,
            speaker=speaker,
            start_structure=start_structure,
            instruction_text=instruction_text,
        )

    # -- unknown / retry --------------------------------------------------
    return ParsedMessage(msg_type=MessageType.UNKNOWN, raw=text)


# ---------------------------------------------------------------------------
# Session state  (one per A2A context_id ≈ one full evaluation run)
# ---------------------------------------------------------------------------

class SessionState:
    """Mutable state for one A2A context (spans multiple seeds)."""

    def __init__(self) -> None:
        self.speaker_profiles: dict[str, SpeakerProfile] = {}
        self.turn_count: int = 0
        self.conversation: list[dict[str, str]] = []   # {"role": ..., "content": ...}
        self.pending: Optional[PendingClassification] = None
        self.current_speaker: Optional[str] = None

    def get_or_create_speaker(self, name: str) -> SpeakerProfile:
        if name not in self.speaker_profiles:
            self.speaker_profiles[name] = SpeakerProfile(name=name)
        return self.speaker_profiles[name]

    def all_speaker_summaries(self) -> str:
        if not self.speaker_profiles:
            return "No speaker conventions observed yet."
        return "\n\n".join(p.summary() for p in self.speaker_profiles.values())

    def reset_for_new_seed(self) -> None:
        """Reset for a new game seed (speakers change, conventions reset)."""
        self.speaker_profiles.clear()
        self.turn_count = 0
        self.pending = None
        self.current_speaker = None
        # Keep conversation for LLM context but trim to last few entries
        # to avoid token explosion over 8 seeds.
        if len(self.conversation) > 10:
            self.conversation = self.conversation[-6:]


# ---------------------------------------------------------------------------
# State manager (across all contexts)
# ---------------------------------------------------------------------------

class StateManager:
    """Registry of session states keyed by A2A context_id."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, context_id: str) -> SessionState:
        if context_id not in self._sessions:
            self._sessions[context_id] = SessionState()
        return self._sessions[context_id]
