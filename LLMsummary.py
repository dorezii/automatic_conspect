from __future__ import annotations

import os
from pathlib import Path

import requests

# Поддерживаем только две целевые instruct-модели по запросу.
MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
DEFAULT_MODEL = "qwen"


def _resolve_model(model: str | None) -> str:
    if not model:
        return MODELS[DEFAULT_MODEL]

    key = model.strip().lower()
    if key in MODELS:
        return MODELS[key]

    # Разрешаем передавать полный model id, но проверяем, что это Qwen/Mistral.
    lowered = model.lower()
    if "qwen" in lowered or "mistral" in lowered:
        return model

    return MODELS[DEFAULT_MODEL]


def summarize_with_llm(text: str, model: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not text.strip():
        return ""

    if not api_key:
        return fallback_summary(text)

    prompt = (
        "Сделай абстрактивный конспект лекции на русском языке. "
        "Структура: тема, ключевые идеи, важные определения/формулы, выводы."
    )
    response = requests.post(
        f"{api_base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": _resolve_model(model),
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Ты помощник для создания учебных конспектов."},
                {"role": "user", "content": f"{prompt}\n\n{text[:20000]}"},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def fallback_summary(text: str, max_lines: int = 12) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    top = lines[:max_lines]
    bullets = "\n".join(f"- {line}" for line in top)
    return "## Черновик конспекта\n\n" + bullets


def save_summary(summary: str, out_path: Path) -> Path:
    out_path.write_text(summary, encoding="utf-8")
    return out_path
