# # Абстрактивная суммаризация транскриптов видеолекций с помощью LLM (text-only)
# 
# Данный ноутбук реализует текстоориентированный конвейер **«транскрипт → конспект»** и поддерживает:
# - предобработку транскрипта;
# - обработку длинного текста через разбиение на фрагменты и агрегацию (map–reduce);
# - унифицированный запуск нескольких LLM-моделей;
# - сбор метрик времени и пикового потребления VRAM (при наличии GPU).
# 
# На текущем этапе в качестве входа используется **только текстовый транскрипт**.
# 

# !pip -q install -U transformers accelerate bitsandbytes sentencepiece pandas yake

import os
import re
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jinja2.exceptions import TemplateError

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

hf_acces_token = "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"


def preprocess_transcript(text: str) -> str:
    """Нормализация транскрипта (без изменения смысла)."""
    if text is None:
        return ""
    # убрать неразрывные пробелы
    text = text.replace(" ", " ")
    # убрать повторяющиеся пробелы/табы
    text = re.sub(r"[ 	]+", " ", text)
    # схлопнуть слишком длинные пустые блоки
    text = re.sub(r"\n{3,}", "\n\n", text)
    # убрать пробелы около переводов строк
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text.lower()))

# Загрузка транскрипта
# Укажите путь к вашему файлу (txt). По умолчанию ищем transcript.txt рядом с ноутбуком.

TRANSCRIPT_PATH = Path("D:\\Magistratura\\3\\НИРС\\Протокол_HTTP_Компьютерные_сети_2024_10.txt")

if not TRANSCRIPT_PATH.exists():
    print(
        f"Файл {TRANSCRIPT_PATH} не найден.\n"
        "Создайте transcript.txt или задайте TRANSCRIPT_PATH на ваш .txt файл."
    )
    source_text = ""
else:
    source_text = TRANSCRIPT_PATH.read_text(encoding="utf-8")

source_text = preprocess_transcript(source_text)
print("Words in transcript:", word_count(source_text))


# In[ ]:


# Реестр моделей (можно расширять)
MODELS: Dict[str, str] = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    # "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma-2-2B-IT": "google/gemma-2-2b-it",
}

USE_4BIT = True  # для GPU обычно выгодно; на CPU может быть недоступно

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# In[ ]:


@dataclass
class ModelLoadConfig:
    use_4bit: bool = True
    torch_dtype: torch.dtype = torch.float16


class HuggingFaceLLM:
    """Уровень инференса: загрузка модели и генерация текста."""

    def __init__(self, model_id: str, load_cfg: ModelLoadConfig):
        self.model_id = model_id
        self.load_cfg = load_cfg
        self.tokenizer = None
        self.model = None

    def load(self):
        # Настройки 4-bit (если доступно)
        quant_cfg = None
        if self.load_cfg.use_4bit and torch.cuda.is_available():
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.load_cfg.torch_dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=(False if 'gemma' in self.model_id.lower() else True), revision="main", use_auth_token=hf_acces_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quant_cfg,
            torch_dtype=self.load_cfg.torch_dtype if torch.cuda.is_available() else None,
            revision="main",
            token=hf_acces_token,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        return self

    def unload(self):
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def build_chat_prompt(self, system: str, user: str) -> str:
        """Собирает чат-промпт через chat_template, если он доступен.

        Некоторые модели (например, Gemma-IT) не поддерживают роль `system`
        в chat_template. В этом случае системная инструкция встраивается
        в первое пользовательское сообщение.
        """
        # Некоторые шаблоны (в т.ч. Gemma-IT) не принимают роль system.
        if "gemma" in self.model_id.lower():
            merged_user = f"{system}\n\n{user}"
            messages = [{"role": "user", "content": merged_user}]
        else:
            # Базовый вариант: system + user
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]


        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TemplateError, ValueError) as e:
                # Fallback для шаблонов без роли system (частая причина: "System role not supported")
                err = str(e).lower()
                if ("system role" in err) or ("role" in err and "system" in err):
                    merged_user = f"{system}\n\n{user}"
                    messages2 = [{"role": "user", "content": merged_user}]
                    return self.tokenizer.apply_chat_template(
                        messages2,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                # Если ошибка иная — падаем на простую разметку
            except Exception:
                # На всякий случай: если шаблон сломан/несовместим
                pass

        # Универсальный текстовый fallback (работает для моделей без chat_template)
        return f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"

    @torch.inference_mode()
    def generate(self, prompt_text: str, max_new_tokens: int = 320, deterministic: bool = True):
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Для конспектирования предпочтительна детерминированность.
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if deterministic:
            gen_kwargs.update(dict(do_sample=False))
        else:
            gen_kwargs.update(dict(do_sample=True, temperature=0.2, top_p=0.9))

        out = self.model.generate(**inputs, **gen_kwargs)
        # отрезаем промпт
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()


# In[ ]:


@dataclass
class SummarizationConfig:
    chunk_tokens: int = 2500
    overlap_tokens: int = 200
    map_max_new_tokens: int = 400
    reduce_max_new_tokens: int = 700
    deterministic: bool = True


def split_by_tokens(tokenizer, text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    """Разбиение длинного транскрипта на фрагменты по токенам модели.

    Важно: для длинных лекций транскрипт часто не помещается в контекст. Поэтому
    применяем chunking + overlap, а затем агрегацию результатов.
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return []

    chunk_tokens = max(128, int(chunk_tokens))
    overlap_tokens = max(0, int(overlap_tokens))
    if overlap_tokens >= chunk_tokens:
        overlap_tokens = max(0, chunk_tokens // 4)

    chunks: List[str] = []
    i = 0
    n = len(ids)
    while i < n:
        j = min(i + chunk_tokens, n)
        chunk_ids = ids[i:j]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if j >= n:
            break
        i = j - overlap_tokens
    return chunks


class MapReduceSummarizer:
    """Прикладной уровень: «транскрипт → конспект» через map–reduce."""

    def __init__(self, cfg: SummarizationConfig):
        self.cfg = cfg
        self.system_prompt = (
            "Ты аккуратный помощник. Делай конспект строго по исходному тексту, без выдумок. "
            "Пиши по-русски. Если факт не указан — не добавляй."
        )

    def map_prompt(self, chunk_text: str) -> str:
        return f"""Сделай конспект фрагмента лекции.

Требования:
- 6–10 пунктов
- без повторов
- сохраняй термины, числа, версии и обозначения
- не добавляй информацию от себя

Формат: Markdown-список.

ФРАГМЕНТ ТРАНСКРИПТА:
{chunk_text}
"""

    def reduce_prompt(self, partials: str) -> str:
        return f"""На основе частичных конспектов сделай единый итоговый конспект лекции.

Требования:
- в начале 1–2 строки: тема и суть
- далее 10–16 пунктов по содержанию
- отдельным блоком выдели термины/определения (только если они есть в транскрипте)
- не добавляй факты, которых нет в исходном тексте

Формат: Markdown.

ЧАСТИЧНЫЕ КОНСПЕКТЫ:
{partials}
"""

    def run(self, llm: HuggingFaceLLM, transcript_text: str) -> Dict[str, object]:
        chunks = split_by_tokens(llm.tokenizer, transcript_text, self.cfg.chunk_tokens, self.cfg.overlap_tokens)
        partials_list: List[str] = []

        # MAP
        for idx, ch in enumerate(chunks, 1):
            user = self.map_prompt(ch)
            prompt = llm.build_chat_prompt(self.system_prompt, user)
            part = llm.generate(
                prompt,
                max_new_tokens=self.cfg.map_max_new_tokens,
                deterministic=self.cfg.deterministic,
            )
            partials_list.append(part)

        # REDUCE
        partials = "\n\n".join(partials_list)
        user2 = self.reduce_prompt(partials)
        prompt2 = llm.build_chat_prompt(self.system_prompt, user2)
        final = llm.generate(
            prompt2,
            max_new_tokens=self.cfg.reduce_max_new_tokens,
            deterministic=self.cfg.deterministic,
        )

        return {"final": final, "partials": partials_list, "n_chunks": len(chunks)}


# In[ ]:


def run_benchmark(model_name: str, model_id: str, transcript_text: str, sum_cfg: SummarizationConfig):
    """Запуск суммаризации + сбор времени и peak VRAM."""
    if not transcript_text.strip():
        return {
            "model": model_name,
            "model_id": model_id,
            "time_summarize_sec": None,
            "peak_vram_mb": None,
            "n_chunks": 0,
            "summary": "",
        }

    # очистка памяти перед запуском
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    llm = HuggingFaceLLM(model_id, ModelLoadConfig(use_4bit=USE_4BIT)).load()
    summarizer = MapReduceSummarizer(sum_cfg)

    t0 = time.perf_counter()
    result = summarizer.run(llm, transcript_text)
    t1 = time.perf_counter()

    peak_mb = None
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    llm.unload()

    return {
        "model": model_name,
        "model_id": model_id,
        "time_summarize_sec": round(t1 - t0, 2),
        "peak_vram_mb": None if peak_mb is None else round(peak_mb, 1),
        "n_chunks": result["n_chunks"],
        "summary": result["final"],
        "partials": result["partials"],
    }


sum_cfg = SummarizationConfig(
    chunk_tokens=2500,
    overlap_tokens=200,
    map_max_new_tokens=420,
    reduce_max_new_tokens=900,
    deterministic=True,
)

runs = []
for name, mid in MODELS.items():
    print()
    print(f"=== Running: {name} ===")
    out = run_benchmark(name, mid, source_text, sum_cfg)
    runs.append(out)
    print(out["summary"][:600] + "...")

# Краткая сводка
pd.DataFrame(runs)[["model", "time_summarize_sec", "peak_vram_mb", "n_chunks"]]


# In[ ]:


# Доп. метрики текста (простые и воспроизводимые)

def basic_stats(source: str, summary: str):
    src_words = len(source.split())
    sum_words = len(summary.split())
    return {
        "src_words": src_words,
        "sum_words": sum_words,
        "compression_x": round(src_words / max(sum_words, 1), 2),
        "sum_chars": len(summary),
    }

rows = []
for r in runs:
    s = basic_stats(source_text, r["summary"])
    rows.append({"model": r["model"], **s})

df_stats = pd.DataFrame(rows).sort_values("compression_x", ascending=False)
df_stats


# In[ ]:


# Keyword recall по YAKE (опционально): оценивает, насколько конспект сохраняет ключевые слова исходника.
# Для ru используем lan="ru".

# !pip -q install yake
import yake

kw_extractor = yake.KeywordExtractor(lan="ru", n=1, top=40)
keywords = [k for k, _ in kw_extractor.extract_keywords(source_text)]


def keyword_recall(summary: str, keywords: List[str]):
    s = summary.lower()
    hit = sum(1 for k in keywords if k.lower() in s)
    total = len(keywords)
    return hit, total, (hit / total if total else 0.0)

rows = []
for r in runs:
    hit, total, rec = keyword_recall(r["summary"], keywords)
    rows.append({"model": r["model"], "kw_hit": hit, "kw_total": total, "kw_recall": round(rec, 3)})

pd.DataFrame(rows).sort_values("kw_recall", ascending=False)


# In[ ]:


# Сохранение результатов
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

# Сводная таблица
pd.DataFrame(runs)[["model", "model_id", "time_summarize_sec", "peak_vram_mb", "n_chunks"]].to_csv(
    out_dir / "runs_summary.csv", index=False
)

# Тексты конспектов
for r in runs:
    p = out_dir / f"summary_{r['model'].replace('/', '_')}.md"
    p.write_text(r["summary"], encoding="utf-8")

print("Saved to:", out_dir.resolve())

