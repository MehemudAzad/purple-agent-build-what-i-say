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
_SPEAKER_RE = re.compile(r"\[SPEAKER\][ \t]*(\S+)")
_START_STRUCT_RE = re.compile(r"\[START_STRUCTURE\][ \t]*(.*?)(?:\n|$)")

# Target structure from feedback
_CORRECT_TARGET_RE = re.compile(r"points\.\s*(.+?)\s*\|")
_INCORRECT_TARGET_RE = re.compile(r"Expected:\s*(.+),\s+but got:")


# ---------------------------------------------------------------------------
# Speaker profile
# ---------------------------------------------------------------------------

@dataclass
class SpeakerProfile:
    """Convention tracker for one speaker within a game session.

    Tracks *conventions* (relationships), not literal values.
    Each observation is one of:
      - "same_as_context"  → the fill matched a color/count already in the instruction
      - "different"        → the fill was a new color/count not mentioned in the instruction

    A consistent speaker (Speaker A) always produces "same_as_context".
    An inconsistent speaker (Speaker B) produces a mix → never reaches confidence.
    """

    name: str
    color_conventions: list[str] = field(default_factory=list)   # "same_as_context" | "different"
    count_conventions: list[str] = field(default_factory=list)   # "same_as_context" | "different"
    turns_seen: int = 0
    correct_builds: int = 0
    incorrect_builds: int = 0
    
    # Lockout flags to permanently mark a speaker as inconsistent once their confidence drops below threshold
    is_unreliable_color: bool = False
    is_unreliable_count: bool = False

    # -- inference ----------------------------------------------------------
    # Minimum number of observations and minimum confidence ratio required
    # before we stop asking and start inferring from the profile.
    _MIN_SAMPLES: int = 3
    _MIN_CONFIDENCE: float = 0.67  # most-common must be ≥ 67% of observations

    # Convention constants
    SAME_AS_CONTEXT = "same_as_context"
    DIFFERENT = "different"

    def add_color_convention(self, convention: str):
        if self.is_unreliable_color:
            return # Locked out, don't even bother tracking anymore
        self.color_conventions.append(convention)
        # Check if they should be locked out
        if len(self.color_conventions) >= self._MIN_SAMPLES:
            counter = Counter(self.color_conventions)
            top_count = counter.most_common(1)[0][1]
            if top_count / len(self.color_conventions) < self._MIN_CONFIDENCE:
                self.is_unreliable_color = True

    def add_count_convention(self, convention: str):
        if self.is_unreliable_count:
            return # Locked out
        self.count_conventions.append(convention)
        # Check if they should be locked out
        if len(self.count_conventions) >= self._MIN_SAMPLES:
            counter = Counter(self.count_conventions)
            top_count = counter.most_common(1)[0][1]
            if top_count / len(self.count_conventions) < self._MIN_CONFIDENCE:
                self.is_unreliable_count = True

    def inferred_color_convention(self) -> Optional[str]:
        """Inferred color convention — only if we have enough confident evidence.

        Returns "same_as_context" if the speaker reliably fills with the
        instruction's context color.  Returns None if data is insufficient
        or split (meaning we should keep asking).
        """
        if self.is_unreliable_color:
            return None
        if len(self.color_conventions) < self._MIN_SAMPLES:
            return None
        counter = Counter(self.color_conventions)
        top_convention, top_count = counter.most_common(1)[0]
        if top_count / len(self.color_conventions) < self._MIN_CONFIDENCE:
            # We also set unreliable here just in case, though add_color_convention handles it
            self.is_unreliable_color = True
            return None  # too ambiguous — keep asking
        # Only auto-resolve if the pattern is "same_as_context".
        if top_convention == self.SAME_AS_CONTEXT:
            return self.SAME_AS_CONTEXT
        return None

    def inferred_count_convention(self) -> Optional[str]:
        """Inferred count convention — only if we have enough confident evidence.

        Returns "same_as_context" if the speaker reliably fills with the
        instruction's context count.  Returns None otherwise.
        """
        if self.is_unreliable_count:
            return None
        if len(self.count_conventions) < self._MIN_SAMPLES:
            return None
        counter = Counter(self.count_conventions)
        top_convention, top_count = counter.most_common(1)[0]
        if top_count / len(self.count_conventions) < self._MIN_CONFIDENCE:
            self.is_unreliable_count = True
            return None
        if top_convention == self.SAME_AS_CONTEXT:
            return self.SAME_AS_CONTEXT
        return None

    # -- serialisation for prompt injection ---------------------------------

    def summary(self) -> str:
        lines = [
            f"Speaker '{self.name}' "
            f"({self.turns_seen} turns, {self.correct_builds} correct, "
            f"{self.incorrect_builds} incorrect):"
        ]
        if self.color_conventions:
            counter = Counter(self.color_conventions)
            inferred = self.inferred_color_convention()
            lines.append(
                f"  Color convention: {dict(counter)} "
                f"→ {'PREDICTABLE (same as context)' if inferred else 'UNPREDICTABLE (keep asking)'}"
            )
        if self.count_conventions:
            counter = Counter(self.count_conventions)
            inferred = self.inferred_count_convention()
            lines.append(
                f"  Count convention: {dict(counter)} "
                f"→ {'PREDICTABLE (same as context)' if inferred else 'UNPREDICTABLE (keep asking)'}"
            )
        if not self.color_conventions and not self.count_conventions:
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
    was_asked: bool = False


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
