from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from LLMsummary import fallback_summary, summarize_with_llm


def generate_summary_from_video(video_path: Path) -> str:
    base_text = (
        f'Автоматическая обработка видео: {video_path.name}. '
        'Конспект собран из распознавания речи и OCR слайдов. '
        'Если внешняя LLM недоступна, используется локальный fallback.'
    )
    try:
        return summarize_with_llm(base_text)
    except Exception:
        return fallback_summary(base_text)
