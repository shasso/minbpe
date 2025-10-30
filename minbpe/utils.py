"""Simple helpers for inspecting and displaying RTL (e.g. Syriac) strings.

These helpers avoid adding required dependencies. If you want proper
visual reordering for display in an LTR terminal, install `python-bidi`
and use `get_display` via `display_for_console(..., use_bidi=True)`.
"""
from typing import List


def codepoints(s: str) -> List[str]:
    """Return a list of hex codepoint strings for characters in s."""
    return [hex(ord(c)) for c in s]


def wrap_rle(s: str) -> str:
    """Wrap the string in Right-to-Left Embedding (RLE) and Pop (PDF).

    This often forces correct visual RTL display when printed inside an
    otherwise left-to-right context (simple and dependency-free).
    """
    RLE = "\u202B"
    PDF = "\u202C"
    return RLE + s + PDF


def add_rlm(s: str) -> str:
    """Prepend a Right-to-Left Mark (RLM) to hint the rendering direction."""
    RLM = "\u200F"
    return RLM + s


def display_for_console(s: str, use_bidi: bool = False) -> str:
    """Return a string suitable for display in an LTR console.

    - If use_bidi=True and python-bidi is installed, uses its get_display
      algorithm to produce a visually-correct LTR rendering.
    - Otherwise falls back to wrapping with RLE/PDF which helps many
      terminals and environments.
    """
    if use_bidi:
        try:
            from bidi.algorithm import get_display
        except Exception:
            # fallback to embedding if python-bidi not available
            return wrap_rle(s)
        return get_display(s)
    else:
        return wrap_rle(s)
