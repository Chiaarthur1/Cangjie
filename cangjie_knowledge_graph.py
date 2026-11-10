"""
Project Cangjie — явный граф знаний с именованными отношениями [v10.0].

Назначение: дать палеографу grounded-контекст по набору компонентов —
семантические поля, фонетические семьи, чтения, древнекитайские реконструкции
и глоссы — со ССЫЛКОЙ на каждое ребро (источник данных). Это защита от
галлюцинаций: LLM получает не «подумай, что это», а список фактов и рассуждает
строго по ним.

Реализация переиспользует уже загруженную онтологию CangjieMeaningDeducer
(ram-кэш: 102 999 знаков, 98 725 IDS, 48 831 глосс, 94 854 сэшэн), поэтому
не парсит сырые файлы заново.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from cangjie_deducer import CangjieMeaningDeducer


def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


class CangjieKnowledgeGraph:
    def __init__(self, data_dir: str = "./data", deducer: Optional[CangjieMeaningDeducer] = None):
        self.data_dir = os.path.abspath(data_dir)
        self.deducer = deducer if deducer is not None else CangjieMeaningDeducer(data_dir=data_dir)

    # --- статистика (реальные счётчики, не константы) -------------------
    def graph_stats(self) -> Dict[str, int]:
        d = self.deducer
        return {
            "nodes": len(d.engine.characters_db) + len(d.radical_map),
            "edges_consists_of": len(d.ids_to_char),
            "edges_reading": sum(1 for g in d.gloss_map.values() if g.get("pinyin")),
            "edges_gloss": len(d.gloss_map),
            "edges_old_chinese": len(d.transitive_baxter),
            "edges_phonetic_family": len(d.component_precedents),
        }

    # --- grounded-контекст ----------------------------------------------
    def context_for_components(self, components: List[str], max_evidence: int = 40) -> Dict[str, Any]:
        comps = [c.strip() for c in components if c and c.strip()]
        semantic_fields: List[str] = []
        phonetic_family: List[str] = []
        readings: List[str] = []
        old_chinese: List[str] = []
        glosses: List[str] = []
        evidence: List[Dict[str, str]] = []

        for comp in comps:
            rm = self.deducer.radical_map.get(comp)
            if rm:
                semantic_fields.append(f"{rm['eng']} (Канси №{rm['id']})")
                evidence.append({
                    "relation": "СЕМАНТИКА_РАДИКАЛА", "component": comp,
                    "value": rm["eng"], "source": "Kangxi-214",
                })

            gm = self.deducer.gloss_map.get(comp)
            if gm:
                if gm.get("definition"):
                    glosses.append(gm["definition"])
                    evidence.append({
                        "relation": "ГЛАССА", "component": comp,
                        "value": gm["definition"], "source": "Unihan kDefinition",
                    })
                if gm.get("pinyin"):
                    readings.append(gm["pinyin"])
                    evidence.append({
                        "relation": "ЧТЕНИЕ", "component": comp,
                        "value": gm["pinyin"], "source": "Unihan kMandarin",
                    })
                if gm.get("etymology_hint"):
                    semantic_fields.append(gm["etymology_hint"])

            tb = self.deducer.transitive_baxter.get(comp)
            if tb:
                if tb.get("phonetic_root"):
                    phonetic_family.append(tb["phonetic_root"])
                if tb.get("pinyin"):
                    readings.append(tb["pinyin"])
                if tb.get("old_chinese"):
                    old_chinese.append(tb["old_chinese"])
                if tb.get("gloss"):
                    glosses.append(tb["gloss"])
                evidence.append({
                    "relation": "СЭШЭН", "component": comp,
                    "phonetic_root": tb.get("phonetic_root", ""),
                    "old_chinese": tb.get("old_chinese", ""),
                    "middle_chinese": tb.get("middle_chinese", ""),
                    "source": "Baxter-Sagart",
                })

            for p in self.deducer.component_precedents.get(comp, []):
                if len(evidence) >= max_evidence:
                    break
                evidence.append({
                    "relation": "ПРЕЦЕДЕНТ", "component": comp,
                    "char": p.get("char", ""), "ids": p.get("ids", ""),
                    "pinyin": p.get("pinyin", ""), "old_chinese": p.get("old_chinese", ""),
                    "gloss": p.get("gloss", ""), "source": "CHISE IDS",
                })
                if p.get("pinyin"):
                    readings.append(p["pinyin"])
                if p.get("old_chinese"):
                    old_chinese.append(p["old_chinese"])

        return {
            "semantic_fields": _dedup(semantic_fields),
            "phonetic_family": _dedup(phonetic_family),
            "readings": _dedup(readings),
            "old_chinese": _dedup(old_chinese),
            "glosses": _dedup(glosses),
            "evidence": evidence[:max_evidence],
            "note": "Все факты выше взяты из knowledge graph. За пределами этого списка гипотез строить нельзя.",
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Project Cangjie — grounded-контекст из графа знаний")
    ap.add_argument("components", nargs="+", help="Компоненты (иероглифы), напр. 辶 首")
    ap.add_argument("--data-dir", default="./data", help="Папка с данными")
    ap.add_argument("--stats", action="store_true", help="Показать статистику графа")
    args = ap.parse_args()

    kg = CangjieKnowledgeGraph(data_dir=args.data_dir)
    if args.stats:
        print(json.dumps(kg.graph_stats(), ensure_ascii=False, indent=2))
    ctx = kg.context_for_components(args.components)
    print(json.dumps(ctx, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
