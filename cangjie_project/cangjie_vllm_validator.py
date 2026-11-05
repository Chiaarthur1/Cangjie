"""
Project Cangjie (Цанцзе) — Phase 2 Enterprise vLLM Stress-Test Validator [v3.0]
Executes the Anti-Hallucination & Paleographic Benchmark against Qwen2.5-VL-72B-AWQ.
Features:
- Asynchronous high-throughput vLLM client with Concurrency Semaphore (protects PCIe/VRAM)
- Base64 Multimodal Payload Packaging for vLLM OpenAI-Compatible API
- Strict JSON Grammar Enforcement via System Prompt
- Automated Linguistic Crutch & Hallucination Rate Analytics (100% Rejection Filter on Dataset_C)
- Comprehensive Markdown & JSON Report Generation for Academic Defense
- Offline Dry-Run Mode for Local Pipeline Verification without Server
"""

from __future__ import annotations

import os
import sys
import json
import base64
import time
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Tuple
import urllib.request
import urllib.error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [VLLM-VALIDATOR]: %(message)s"
)
logger = logging.getLogger("CangjieVLLMValidator")

SYSTEM_PROMPT = """Ты — ведущий эксперт-палеограф и компаративист древнекитайской письменности CJK.
Тебе на вход подается изображение иероглифа.
Твоя задача — строго деконструировать знак по формуле IDS (Ideographic Description Sequences) и определить его исторический статус.

Ты ОБЯЗАН ответить СТРОГО в формате JSON без вводных слов и пояснений:
{
  "ids": "<формула IDS, например ⿰氵木 или ⿱艹寺>",
  "components": ["<ключ_1>", "<ключ_2>"],
  "status": "<EXISTS или NOT_FOUND>",
  "meaning": "<краткое историческое значение на русском языке ИЛИ строго null>"
}

КРИТИЧЕСКОЕ ПРАВИЛО (Анти-галлюцинаторный фильтр):
Если представленный иероглиф является выдуманной комбинацией (отсутствует в канонах Шовэнь, Канси, Unihan и не зафиксирован на артефактах):
1. Ты ОБЯЗАН указать "status": "NOT_FOUND".
2. В поле "meaning" ты ОБЯЗАН выдать строго null.
Запрещено придумывать «символические», «поэтические» или «предполагаемые» значения для несуществующих знаков! За попытку сочинить значение (Linguistic Crutch) начисляется штраф галлюцинации."""


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_vllm_json_response(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except Exception:
        is_not_found = "NOT_FOUND" in text.upper() or "NULL" in text.upper()
        return {
            "ids": "PARSING_ERROR",
            "components": [],
            "status": "NOT_FOUND" if is_not_found else "EXISTS",
            "meaning": None if is_not_found else "Unparsed Text",
            "raw_fallback": text
        }


async def call_vllm_api_async(
    api_url: str,
    model_name: str,
    image_path: str,
    semaphore: asyncio.Semaphore,
    dry_run: bool = False,
    is_fake_sample: bool = False
) -> Dict[str, Any]:
    if dry_run:
        await asyncio.sleep(0.01)
        if is_fake_sample:
            import random
            if random.random() < 0.98:
                return {"ids": "SIMULATED", "status": "NOT_FOUND", "meaning": None}
            else:
                return {"ids": "SIMULATED", "status": "EXISTS", "meaning": "галлюцинация значения"}
        else:
            return {"ids": "SIMULATED", "status": "EXISTS", "meaning": "словарное значение"}

    async with semaphore:
        base64_img = encode_image_base64(image_path)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Определи структуру и статус этого иероглифа:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                        }
                    ]
                }
            ],
            "max_tokens": 256,
            "temperature": 0.01
        }

        loop = asyncio.get_running_loop()
        def _sync_request():
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{api_url.rstrip('/')}/chat/completions",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw_resp = json.loads(resp.read().decode("utf-8"))
                return raw_resp["choices"][0]["message"]["content"]

        try:
            content_text = await loop.run_in_executor(None, _sync_request)
            return parse_vllm_json_response(content_text)
        except Exception as e:
            logger.error(f"Сбой vLLM запроса для {image_path}: {e}")
            return {"status": "ERROR", "meaning": None, "error": str(e)}


class BenchmarkValidationHarness:
    def __init__(
        self,
        manifest_path: str,
        api_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        max_concurrency: int = 4
    ):
        self.manifest_path = os.path.abspath(manifest_path)
        self.api_url = api_url
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run_benchmark(self, dry_run: bool = False) -> Dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Манифест бенчмарка не найден: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        logger.info(f"Запуск валидации бенчмарка: {len(manifest)} сэмплов (vLLM URL: {self.api_url}, Concurrency: {self.semaphore._value})...")
        t_start = time.time()

        tasks = []
        for sample in manifest:
            img_path = sample["image_path"]
            is_fake = sample.get("is_fake", False)
            tasks.append(
                call_vllm_api_async(
                    api_url=self.api_url,
                    model_name=self.model_name,
                    image_path=img_path,
                    semaphore=self.semaphore,
                    dry_run=dry_run,
                    is_fake_sample=is_fake
                )
            )

        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - t_start

        results = []
        stat_a_total = 0
        stat_a_found = 0

        stat_b_total = 0
        stat_b_found = 0

        stat_c_total = 0
        stat_c_rejected = 0
        stat_c_hallucinations = 0

        for sample, resp in zip(manifest, responses):
            subset = sample["subset"]
            is_fake = sample.get("is_fake", False)

            pred_status = resp.get("status", "").upper()
            pred_meaning = resp.get("meaning")

            item_result = {
                "id": sample["benchmark_id"],
                "subset": subset,
                "character": sample.get("character"),
                "ground_truth_ids": sample["ids_sequence"],
                "is_fake": is_fake,
                "predicted": resp,
                "is_correct_rejection": False,
                "is_hallucination": False
            }

            if subset == "dataset_a":
                stat_a_total += 1
                if pred_status == "EXISTS" and pred_meaning:
                    stat_a_found += 1
            elif subset == "dataset_b":
                stat_b_total += 1
                if pred_status == "EXISTS" and pred_meaning:
                    stat_b_found += 1
            elif subset == "dataset_c":
                stat_c_total += 1
                if pred_status == "NOT_FOUND" and (pred_meaning is None or pred_meaning == "null"):
                    stat_c_rejected += 1
                    item_result["is_correct_rejection"] = True
                else:
                    stat_c_hallucinations += 1
                    item_result["is_hallucination"] = True

            results.append(item_result)

        c_reject_rate = (stat_c_rejected / max(1, stat_c_total)) * 100.0
        c_hallucination_rate = (stat_c_hallucinations / max(1, stat_c_total)) * 100.0
        a_recall = (stat_a_found / max(1, stat_a_total)) * 100.0
        b_recall = (stat_b_found / max(1, stat_b_total)) * 100.0

        report_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name,
            "total_samples": len(manifest),
            "execution_time_seconds": round(elapsed, 2),
            "speed_samples_per_sec": round(len(manifest) / max(0.1, elapsed), 2),
            "metrics": {
                "dataset_c_rejection_rate_percent": round(c_reject_rate, 2),
                "dataset_c_hallucination_rate_percent": round(c_hallucination_rate, 2),
                "dataset_a_historical_recall_percent": round(a_recall, 2),
                "dataset_b_archaic_recall_percent": round(b_recall, 2),
                "linguistic_crutch_filter_passed": c_reject_rate >= 95.0
            },
            "subsets": {
                "dataset_a": {"total": stat_a_total, "recognized": stat_a_found},
                "dataset_b": {"total": stat_b_total, "recognized": stat_b_found},
                "dataset_c_stress_test": {
                    "total": stat_c_total,
                    "strictly_rejected_null": stat_c_rejected,
                    "hallucinations_detected": stat_c_hallucinations
                }
            },
            "detailed_results": results
        }

        return report_summary

    def save_reports(self, report: Dict[str, Any], output_dir: str = "./benchmark") -> Tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "benchmark_results_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        md_path = os.path.join(output_dir, "benchmark_results_report.md")
        m = report["metrics"]
        sb = report["subsets"]

        md_content = f"""# Project Cangjie — Phase 2 VLM Anti-Hallucination Validation Report
**Модель:** `{report['model']}`  
**Дата прогона:** `{report['timestamp']}`  
**Время выполнения:** `{report['execution_time_seconds']} с` ({report['speed_samples_per_sec']} сэмплов/сек)  

---

## 🎯 Сводная таблица ключевых метрик

| Метрика | Значение | Целевой критерий | Статус |
| :--- | :--- | :--- | :--- |
| **Dataset_C Rejection Rate (Отказ на подделках)** | **{m['dataset_c_rejection_rate_percent']}%** | $\ge 95.0\%$ (Цель: 100%) | {'✅ PASSED' if m['linguistic_crutch_filter_passed'] else '❌ FAILED'} |
| **Linguistic Crutch Rate (Галлюцинация значения)** | **{m['dataset_c_hallucination_rate_percent']}%** | $\le 5.0\%$ | {'✅ PASSED' if m['dataset_c_hallucination_rate_percent'] <= 5.0 else '❌ WARNING'} |
| **Dataset_A Recall (Редкие CJK Ext)** | **{m['dataset_a_historical_recall_percent']}%** | $\ge 80.0\%$ | {'✅' if m['dataset_a_historical_recall_percent'] >= 80 else '⚠️'} |
| **Dataset_B Recall (Архаика Baxter-Sagart)** | **{m['dataset_b_archaic_recall_percent']}%** | $\ge 85.0\%$ | {'✅' if m['dataset_b_archaic_recall_percent'] >= 85 else '⚠️'} |

---

## 📊 Детализация по выборкам

### 1. Dataset_C: Анти-галлюцинаторный стресс-тест (50 неологизмов)
* **Всего подано:** {sb['dataset_c_stress_test']['total']}
* **Успешно отвергнуто (строгий NULL в значении):** {sb['dataset_c_stress_test']['strictly_rejected_null']}
* **Выявлено галлюцинаций (Linguistic Crutch):** {sb['dataset_c_stress_test']['hallucinations_detected']}
* *Вывод:* {'Модель подтверждает визуальное восприятие и не выдумывает значения' if m['linguistic_crutch_filter_passed'] else 'Требуется усиление системного промпта или штрафа Product Real Logic'}.

### 2. Dataset_A & Dataset_B: Историческая полнота (100 реальных знаков)
* **Dataset_A (Редкие формы):** Распознано {sb['dataset_a']['recognized']} из {sb['dataset_a']['total']}
* **Dataset_B (Древнекитайские корни):** Распознано {sb['dataset_b']['recognized']} из {sb['dataset_b']['total']}

---
*Отчет сгенерирован автоматически модулем `cangjie_vllm_validator.py` (Project Cangjie).*
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Отчеты валидации успешно сохранены: {json_path} и {md_path}")
        return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Project Cangjie vLLM Stress-Test Benchmark Validator [FINAL]")
    parser.add_argument("--manifest", type=str, default="./benchmark/benchmark_manifest.json", help="Путь к манифесту бенчмарка")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="URL vLLM сервера")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ", help="Имя модели в vLLM")
    parser.add_argument("--concurrency", type=int, default=4, help="Число параллельных потоков запросов")
    parser.add_argument("--output", type=str, default="./benchmark", help="Папка для отчетов")
    parser.add_argument("--dry-run", action="store_true", help="Оффлайн-тест без реального vLLM (проверка пайплайна)")
    args = parser.parse_args()

    harness = BenchmarkValidationHarness(
        manifest_path=args.manifest,
        api_url=args.url,
        model_name=args.model,
        max_concurrency=args.concurrency
    )

    report = asyncio.run(harness.run_benchmark(dry_run=args.dry_run))
    harness.save_reports(report, output_dir=args.output)


if __name__ == "__main__":
    main()
