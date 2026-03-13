"""
Prompt templates for the two-stage LLM pipeline.

Stage 1 – Classify & Extract:
    Determines whether the instruction is underspecified (color / count / none),
    extracts the explicitly mentioned colours and counts, and recommends BUILD vs ASK.

Stage 2 – Generate [BUILD]:
    Given the (optionally resolved) instruction + start structure, produces the full
    [BUILD];Color,x,y,z;... response with correct coordinates.

Direct prompt:
    Used for handling answers to [ASK] questions and retry/error messages where we
    need to produce a response from conversational context rather than a fresh instruction.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Classify & Extract
# ═══════════════════════════════════════════════════════════════════════════════

STAGE1_SYSTEM = """\
You are an instruction analyst for a 3-D block-building game on a 9x9 grid.

GRID KNOWLEDGE:
- The grid is a 9x9 square in the x-z plane.
- It has 4 corners at: (-400, 400), (400, 400), (400, -400), (-400, -400).
- The origin (0,0) is the center.

Your task: given a natural-language building instruction, determine whether any
information is UNDERSPECIFIED (missing colour or missing block count).

DEFINITIONS:
- "color": The instruction tells you to place blocks but does NOT state their
  colour and you cannot infer it from the speaker's history.
- "count": The instruction names a colour but does NOT state how many blocks.
- "none": Both colour AND count are clear or inferable.

CRITICAL RULES FOR ASKING:
1. ONLY suggest a question if "ambiguity" is "color" or "count".
2. DO NOT ASK about grid properties (e.g., "how many corners", "how big is the grid").
3. DO NOT ASK about game rules, scoring, or the benchmark.
4. DO NOT ASK for repetition if the instruction is clear but complex.
5. Keep suggested questions extremely short and focused on ONLY the missing color or count.

RESPONSE FORMAT — reply with ONLY valid JSON:
{
  "ambiguity": "none" | "color" | "count",
  "missing_description": "<which blocks are missing what, or empty string>",
  "explicitly_mentioned_colors": ["Purple", "Green"],
  "explicitly_mentioned_counts": {"Purple": 5, "Green": 3},
  "suggested_question": "<one short clarifying question ONLY if color/count is missing>",
  "confidence_in_build": <float 0.0–1.0>,
  "reasoning": "<one sentence explaining your decision>"
}
"""


def stage1_user_prompt(
    instruction: str,
    start_structure: str,
    speaker_name: str,
    speaker_summary: str,
) -> str:
    """Build the user message for the Stage-1 classification call."""
    return (
        f"SPEAKER: {speaker_name}\n"
        f"SPEAKER HISTORY:\n{speaker_summary}\n\n"
        f"START_STRUCTURE: {start_structure or '(empty grid)'}\n\n"
        f"INSTRUCTION: {instruction}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Generate [BUILD]
# ═══════════════════════════════════════════════════════════════════════════════

STAGE2_SYSTEM = """\
You are a precise block-building agent on a 9×9 3-D grid.

COORDINATE SYSTEM
-----------------
X (left–right):  -400, -300, -200, -100, 0, 100, 200, 300, 400
Z (front–back):  -400, -300, -200, -100, 0, 100, 200, 300, 400
Y (height):      50 = ground, 150, 250, 350, 450

SPATIAL ORIENTATION
-------------------
"In front"  → +Z (e.g. if starting at Z=0, in front is Z=100)
"Behind"    → -Z (e.g. if starting at Z=0, behind is Z=-100)
"Right"     → +X
"Left"      → -X

STACK VS ROW
------------
"Stack", "Tower", "On top" → Vertical alignment. Change ONLY the Y coordinate.
"Row", "Line", "Beside"   → Horizontal alignment. Change ONLY the X or Z coordinate.

PLACEMENT RULES
---------------
1. Relative Placement: "Build X in front of Y" means X must have the SAME X-coordinate as Y, and an increased Z-coordinate. 
2. Stacking: A stack of 3 blocks at (0,0) means: (0, 50, 0), (0, 150, 0), (0, 250, 0).
3. Grid Limits: Never use coordinates outside [-400, 400].

OUTPUT FORMAT
-------------
Think step-by-step:
1. Parse every block in the START_STRUCTURE.
2. Calculate NEW coordinates based on spatial directions.
3. Verify "Stack" (Y-axis) vs "Row" (X/Z-axis) usage.
4. Output one line: `[BUILD];Color,x,y,z;Color,x,y,z;...`

CRITICAL: Include ALL blocks (existing + new). Use `Color,x,y,z;` format precisely.
"""


def stage2_user_prompt(
    instruction: str,
    start_structure: str,
    resolution: str | None = None,
) -> str:
    """Build the user message for the Stage-2 build-generation call."""
    parts: list[str] = []
    parts.append(f"START_STRUCTURE: {start_structure or '(empty grid)'}")
    if resolution:
        parts.append(f"RESOLVED AMBIGUITY: {resolution}")
    parts.append(f"INSTRUCTION: {instruction}")
    parts.append("\nRespond with the complete [BUILD] string.")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Direct / fallback prompt  (for answers, retries, error recovery)
# ═══════════════════════════════════════════════════════════════════════════════

DIRECT_SYSTEM = """\
You are a block-building agent.  You previously attempted to build a structure
and either asked a clarifying question (which has now been answered) or produced
an invalid response (which you need to correct).

Use the conversation history below to produce your response.

READING THE ANSWER TO YOUR QUESTION
-------------------------------------
The answer will be SHORT and DIRECT. Examples:
  - "4 blocks" or "3"           → use that exact count
  - "Red and Blue"               → those are the colors to use
  - "Colors in target: Blue, Blue, Green" → count repeats: 2 Blue + 1 Green
Use the answer to fill in the missing color or count, then produce [BUILD].

COORDINATE RULES
-----------------
Valid X, Z: -400, -300, -200, -100, 0, 100, 200, 300, 400
Valid Y:    50, 150, 250, 350, 450   (ground = 50, each stacked block adds 100)
Format:     Color,x,y,z  — colour capitalised, no spaces

Respond with EXACTLY one of the following formats:

Format A (if building):
[BUILD];Color,x,y,z;Color,x,y,z;...

Format B (if you MUST ask a question instead):
[ASK];your question

Include ALL blocks that should be on the grid (existing + new).
Do NOT output any other text.
"""
