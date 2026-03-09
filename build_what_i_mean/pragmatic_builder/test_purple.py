"""Quick smoke tests for validators + state modules."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "purple_agent"))

from validators import validate_build_response, validate_block, snap, VALID_XZ, VALID_Y
from state import parse_message, MessageType, SessionState, SpeakerProfile

# === Validators ===
assert snap(55, VALID_Y) == 50
assert snap(120, VALID_Y) == 150
assert snap(-350, VALID_XZ) == -400
print("  snap: OK")

assert validate_block("Red,0,50,0") == "Red,0,50,0"
assert validate_block("red,0,50,0") == "Red,0,50,0"
assert validate_block("Green, -200, 150, 400") == "Green,-200,150,400"
assert validate_block("bad") is None
assert validate_block("") is None
print("  validate_block: OK")

r = validate_build_response("[BUILD];Red,0,50,0;Green,-200,150,400")
assert r == "[BUILD];Red,0,50,0;Green,-200,150,400", r
print("  validate_build_response: OK")

# === State parsing ===
msg = "[TASK_DESCRIPTION] Grid: 9x9 cells.)\n[SPEAKER] Anna\n[START_STRUCTURE] Blue,0,50,0\nPlace a red block on top."
p = parse_message(msg)
assert p.msg_type == MessageType.INSTRUCTION
assert p.speaker == "Anna"
assert p.start_structure == "Blue,0,50,0"
assert "red block" in p.instruction_text
print("  parse instruction: OK")

msg = "Feedback: Correct structure built! +10 points. Red,0,50,0;Blue,0,150,0 | Round score: +10 | Total score: +10"
p = parse_message(msg)
assert p.msg_type == MessageType.FEEDBACK
assert p.is_correct is True
assert p.target_structure == "Red,0,50,0;Blue,0,150,0", repr(p.target_structure)
print("  parse feedback (correct): OK")

msg = "Feedback: Incorrect structure. -10 points. Expected: Red,0,50,0;Blue,0,150,0, but got: Green,0,50,0 | Round score: -10 | Total score: -10"
p = parse_message(msg)
assert p.msg_type == MessageType.FEEDBACK
assert p.is_correct is False
assert p.target_structure == "Red,0,50,0;Blue,0,150,0", repr(p.target_structure)
print("  parse feedback (incorrect): OK")

msg = "Answer: Yellow (-5 points for asking)"
p = parse_message(msg)
assert p.msg_type == MessageType.ANSWER
print("  parse answer: OK")

msg = "A new task is starting, now you will play the game again."
p = parse_message(msg)
assert p.msg_type == MessageType.NEW_TASK
print("  parse new task: OK")

# === Speaker profile ===
sp = SpeakerProfile(name="Anna")
sp.color_fills.extend(["Yellow", "Yellow", "Purple"])
assert sp.inferred_color() == "Yellow"
sp.count_fills.extend([3, 3, 2])
assert sp.inferred_count() == 3
print("  speaker profile: OK")

# === Session reset ===
ss = SessionState()
ss.get_or_create_speaker("Anna").turns_seen = 5
ss.conversation.extend([{"role": "user", "content": "x"}] * 20)
ss.reset_for_new_seed()
assert len(ss.speaker_profiles) == 0
assert len(ss.conversation) <= 6
print("  session reset: OK")

print("\n=== ALL TESTS PASSED ===")
