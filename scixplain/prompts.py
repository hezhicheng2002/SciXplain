from __future__ import annotations

from typing import Dict


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def _context_block(ex_ctx: Dict) -> str:
    paragraph = _clean((ex_ctx or {}).get("paragraph", ""))
    ocr = _clean((ex_ctx or {}).get("ocr", ""))
    parts = []
    if paragraph:
        parts.append(f"Paragraph context: {paragraph}")
    if ocr:
        parts.append(f"OCR tokens: {ocr}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def build_caption_short_prompt(ex_ctx: Dict) -> str:
    return (
        "Write one concise sentence that captions the scientific illustration.\n"
        "Stay grounded in the visible figure.\n"
        f"{_context_block(ex_ctx)}"
        "Answer with the short caption only."
    ).strip()


def build_caption_long_prompt(ex_ctx: Dict) -> str:
    return (
        "Write a detailed caption for the scientific illustration.\n"
        "Summarize the main components, flow, and purpose without inventing content.\n"
        f"{_context_block(ex_ctx)}"
        "Answer with the long caption only."
    ).strip()


def build_description_prompt(ex_ctx: Dict) -> str:
    return (
        "Describe the scientific illustration in a structure-aware way.\n"
        "Mention the main regions, labels, and relations that are visible.\n"
        f"{_context_block(ex_ctx)}"
        "Answer with the description only."
    ).strip()


def build_explanation_prompt(ex_ctx: Dict) -> str:
    return (
        "Explain the process or mechanism shown in the scientific illustration.\n"
        "Ground the explanation in visible structure and the provided context.\n"
        f"{_context_block(ex_ctx)}"
        "Answer with the explanation only."
    ).strip()

