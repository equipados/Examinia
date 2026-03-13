"""Precios de tokens por modelo y utilidades de coste compartidas."""
from __future__ import annotations

# Precios por millón de tokens (USD)
TOKEN_PRICES: dict[str, dict[str, float]] = {
    # Gemini
    "gemini-2.5-pro":   {"input": 1.25,  "output": 10.0},
    "gemini-2.5-flash": {"input": 0.15,  "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10,  "output": 0.40},
    "gemini-1.5-pro":   {"input": 1.25,  "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # OpenAI
    "gpt-4o":           {"input": 2.50,  "output": 10.0},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "o4-mini":          {"input": 1.10,  "output": 4.40},
    "o3-mini":          {"input": 1.10,  "output": 4.40},
    "o1":               {"input": 15.0,  "output": 60.0},
    "o1-mini":          {"input": 3.00,  "output": 12.0},
    # DeepSeek
    "deepseek-chat":      {"input": 0.27,  "output": 1.10},
    "deepseek-reasoner":  {"input": 0.55,  "output": 2.19},
}

USD_TO_EUR = 0.92

# Thinking tokens (Gemini 2.5 cobra el pensamiento interno por separado)
THINKING_PRICES: dict[str, float] = {
    "gemini-2.5-flash": 3.50,   # USD por 1M thinking tokens
    "gemini-2.5-pro":   3.50,
}

OP_LABELS: dict[str, str] = {
    "extract_solutions":        "Extracción IA",
    "extract_solutions_pages":  "Extracción páginas",
    "extract_teacher_pdf":      "PDF profesor",
    "grade_submission":         "Corrección (assess)",
    "grade_submission_solver":  "Corrección (solver)",
}


def provider_of(model: str) -> str:
    """Devuelve el proveedor según el nombre del modelo."""
    if model.startswith("deepseek"):
        return "DeepSeek"
    if model.startswith(("gpt-", "o1", "o3", "o4", "text-")):
        return "OpenAI"
    return "Gemini"


def cost_eur(model: str, inp: int, out: int, thinking: int = 0) -> float:
    prices = next(
        (v for k, v in TOKEN_PRICES.items() if model.startswith(k)),
        {"input": 0.0, "output": 0.0},
    )
    think_price = next(
        (v for k, v in THINKING_PRICES.items() if model.startswith(k)),
        0.0,
    )
    usd = (inp * prices["input"] + out * prices["output"] + thinking * think_price) / 1_000_000
    return usd * USD_TO_EUR
