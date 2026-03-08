"""
Coordinate validation and output format enforcement for block structures.

Ensures all block coordinates snap to valid grid positions and responses
conform to the [BUILD]/[ASK] protocol expected by the green agent.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Valid grid positions
# ---------------------------------------------------------------------------

VALID_XZ: frozenset[int] = frozenset(range(-400, 401, 100))   # -400 … 400 step 100
VALID_Y: frozenset[int] = frozenset(range(50, 501, 100))      # 50, 150, 250, 350, 450

VALID_COLORS: frozenset[str] = frozenset({
    "Red", "Blue", "Green", "Yellow", "Purple", "Orange",
    "White", "Black", "Brown", "Pink", "Gray", "Grey",
})

# Regex for a single block token: Color,int,int,int
_BLOCK_RE = re.compile(r"^([A-Za-z]+),\s*(-?\d+),\s*(\d+),\s*(-?\d+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def snap(value: int, valid: frozenset[int]) -> int:
    """Snap *value* to the nearest member of *valid*."""
    return min(valid, key=lambda v: abs(v - value))


def normalize_color(color: str) -> str:
    """Capitalize a color name (``red`` → ``Red``)."""
    return color.strip().capitalize()


# ---------------------------------------------------------------------------
# Single block
# ---------------------------------------------------------------------------

def validate_block(block_str: str) -> str | None:
    """
    Validate and fix a single ``Color,x,y,z`` token.

    Returns the corrected string, or ``None`` if the token cannot be salvaged.
    """
    block_str = block_str.strip()
    if not block_str:
        return None

    m = _BLOCK_RE.match(block_str)
    if not m:
        return None

    color = normalize_color(m.group(1))
    x = snap(int(m.group(2)), VALID_XZ)
    y = snap(int(m.group(3)), VALID_Y)
    z = snap(int(m.group(4)), VALID_XZ)

    return f"{color},{x},{y},{z}"


# ---------------------------------------------------------------------------
# Full [BUILD] response
# ---------------------------------------------------------------------------

def validate_build_response(response: str) -> str:
    """
    Validate and fix a ``[BUILD]`` response.

    * Strips extraneous text around the ``[BUILD]`` line.
    * Snaps every coordinate to the nearest valid grid position.
    * Drops unparseable block tokens silently.

    Returns the cleaned response string (may still start with ``[BUILD]``
    even if some blocks were dropped).
    """
    response = response.strip()

    # If the model wrapped between markdown code fences, unwrap
    if "```" in response:
        lines = response.splitlines()
        cleaned: list[str] = []
        inside_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                inside_fence = not inside_fence
                continue
            if inside_fence or not line.strip().startswith("```"):
                cleaned.append(line)
        response = "\n".join(cleaned).strip()

    # If multiple lines, find the one that starts with [BUILD]
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("[BUILD]"):
            response = line
            break

    if not response.startswith("[BUILD]"):
        # Try to salvage: if it looks like block coordinates, wrap it.
        if ";" in response and "," in response:
            response = "[BUILD];" + response.lstrip(";")
        else:
            return response  # Can't fix, return as-is

    # Parse blocks after the prefix
    content = response[7:]  # after "[BUILD]"
    if content.startswith(";"):
        content = content[1:]

    blocks = [b.strip() for b in content.split(";") if b.strip()]
    validated: list[str] = []
    for block in blocks:
        fixed = validate_block(block)
        if fixed:
            validated.append(fixed)

    if not validated:
        return response  # Nothing salvageable, return original

    return "[BUILD];" + ";".join(validated)


# ---------------------------------------------------------------------------
# Format check
# ---------------------------------------------------------------------------

def is_valid_response(response: str) -> bool:
    """Return ``True`` if *response* starts with ``[BUILD]`` or ``[ASK]``."""
    return response.startswith("[BUILD]") or response.startswith("[ASK]")
