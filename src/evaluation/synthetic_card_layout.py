"""Lossless layout for new synthetic evidence; historical renderers stay frozen."""

from __future__ import annotations

from pathlib import Path

from scripts.build_exploration_pool_v4 import FONT, HEIGHT, WIDTH, _rect

CARD_FONT = {**FONT, "_": ("00000",) * 6 + ("11111",)}


def layout_card(lines: list[str]) -> list[dict[str, int | str]]:
    """Fit complete lines at readable scale, rejecting unsupported or overflowing text."""
    if not lines or len(lines) > 7:
        raise ValueError("card requires 1 through 7 complete lines")
    layout = []
    for index, line in enumerate(lines):
        text = line.upper()
        if not text or any(char not in CARD_FONT for char in text):
            raise ValueError("empty line or unsupported glyph")
        scale = min(3, (WIDTH - 40) // (6 * len(text)))
        if scale < 2:
            raise ValueError("line cannot fit at minimum readable scale")
        layout.append({"text": text, "x": 20, "y": 27 + index * 30, "scale": scale})
    return layout


def write_card(path: Path, lines: list[str], seed: int) -> list[dict[str, int | str]]:
    layout = layout_card(lines)
    pixels = bytearray([244, 246, 249] * WIDTH * HEIGHT)
    accent = ((seed * 47) % 120 + 60, (seed * 71) % 120 + 60, (seed * 29) % 120 + 60)
    _rect(pixels, 0, 0, WIDTH, 18, accent)
    _rect(pixels, 0, HEIGHT - 14, WIDTH, 14, accent)
    for index, item in enumerate(layout):
        if index % 2:
            _rect(pixels, 12, int(item["y"]) - 5, WIDTH - 24, 26, (231, 235, 241))
        scale = int(item["scale"])
        for character_index, character in enumerate(str(item["text"])):
            for glyph_y, bits in enumerate(CARD_FONT[character]):
                for glyph_x, bit in enumerate(bits):
                    if bit == "1":
                        _rect(pixels, 20 + (character_index * 6 + glyph_x) * scale,
                              int(item["y"]) + glyph_y * scale, scale, scale, (25, 31, 42))
    for index in range(5):
        _rect(pixels, WIDTH - 36 + index * 5, 3 + ((seed + index * 11) % 10), 3, 8, accent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        handle.write(pixels)
    return layout
