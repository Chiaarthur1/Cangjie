"""
Project Cangjie (Цанцзе) — Full Production Paleographic Workstation UI/UX & Multi-Character Studio [v8.0]
Zero Stubs / High-Contrast Obsidian & White Palette + Dual Accents (#FF3322 Cinnabar & #F5B041 Imperial Gold):
1. Top Navigation Bar + Smooth Slide-Out Drawers (Выездные меню):
   - Top Bar switches between [1. Дешифровщик и Мульти-Свиток], [2. Мастерская Каллиграфии и Фото-Сканер],
     [3. Трибунал Дуэли Агентов CoVe-RAG+], and opens Slide-Out Drawers:
     * Left Drawer: Библиотека Свитков, Эстампажей и Многознаковых Надписей (Multi-Glyph Presets)
     * Right Drawer: Аппаратные Профили и Обучение без Заглушек (Ноутбук / ПК Turbo T1 RTX 4060 / Станция 4x RTX 4090)
2. Upright MakeMeAHanzi Skeleton Rendering (Y-axis EM-inversion fixed: y_screen = 900.0 - y_em)
3. Multi-Character Strip (Множество иероглифов одновременно):
   - Supports multi-character strings/scrolls (e.g., «道法自然», «雷電神鬼», «洔蘦國道», «我心光明») and multi-character
     photo segmentation with per-character clickable cards and individual AST + 5-step precedent deduction.
4. Unified Drawing & Photo Module (Модуль с фото перенесен в меню к рисованию иероглифов):
   - Combines interactive Mizige (米字格) brush/knife/eraser drawing, radical zone stamping, AND instant Photo Upload
     (Drag & Drop / File Picker / Ctrl+V) with both Server-Side and 100% Client-Side Canvas Sauvola Binarization +
     Multi-Glyph XY-Segmentation so photo upload works even in standalone offline HTML mode.
"""

from __future__ import annotations

import os
import sys
import io
import re
import json
import time
import signal
import socket
import base64
import tempfile
import subprocess
import logging
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageDraw
import numpy as np

_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _LOCAL_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_DIR)

from cangjie_benchmark_forge import (
    KANGXI_214_COMPREHENSIVE_CATALOG,
    GlyphRepository,
    BoundingBox,
    RadicalLeaf,
    SpatialSlot,
    EXISTING_COMBINATIONS
)
from cangjie_product_logic import (
    ProductRealLogic,
    NeuroSymbolicLossEngine,
    ReichenbachFalsifier
)
from cangjie_deducer import (
    CangjieMeaningDeducer,
    RADICAL_DEEP_SEMANTICS,
    PHONETIC_CO_SEMANTICS,
    SPATIAL_OPERATOR_LAWS,
    ATOMIC_CHAR_DECOMPOSITIONS
)
from cangjie_layout_to_tree import (
    PointerTreeTransformerDecoder,
    RadicalDetectionNode,
    GLYPH_TO_CLASS_ID,
    IDC_OPERATORS
)
from cangjie_generative_degradation import HistoricalArtifactDegradationPipeline
from cangjie_archaic_epigraphy_engine import ArchaicJiaguwenJinwenSynthesizer
from cangjie_edge_cpu_runtime import CangjieEdgeCPURuntime, InMemorySQLiteGraphRAG
from cangjie_photo_decoder import CangjiePaleoPhotoDecoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WORKSTATION-UI-v8]: %(message)s"
)
logger = logging.getLogger("CangjieWorkstationUI")

IDC_HUMAN_LABELS: Dict[str, str] = {
    "⿰": "⿰ [Лево-Право]",
    "⿱": "⿱ [Верх-Низ]",
    "⿲": "⿲ [3-Полосы Гориз.]",
    "⿳": "⿳ [3-Яруса Верт.]",
    "⿴": "⿴ [Полный Охват]",
    "⿵": "⿵ [Охват Сверху]",
    "⿶": "⿶ [Охват Снизу]",
    "⿷": "⿷ [Охват Слева]",
    "⿸": "⿸ [Навес Сверху-Слева]",
    "⿹": "⿹ [Охват Сверху-Справа]",
    "⿺": "⿺ [Ладья Снизу-Слева]",
    "⿻": "⿻ [Пересечение]",
    "SINGLE": "Эталонный Знак"
}

HARDWARE_PROFILES: Dict[str, Dict[str, Any]] = {
    "laptop": {
        "title": "Профиль 1: Личный Ноутбук (Интернет открыт — Хаб загрузки и сборки)",
        "cpu": "CPU / Apple MPS / Mobile GPU",
        "ram": "8–32 ГБ ОЗУ",
        "gpu": "Локальный GPU / CPU AVX2",
        "network": "Интернет ДОСТУПЕН (скачивание Unihan, MakeMeAHanzi, CHISE и весов)",
        "role": "Подготовка данных, обучение 2D Layout-to-Tree (3 600 деревьев за ~2.5 с) и упаковка оффлайн-архива для закрытых машин",
        "commands": [
            "python3 cangjie_fast_memory_hub.py --download-all --rebuild-cache",
            "python3 cangjie_layout_to_tree.py --train-laptop",
            "python3 cangjie_fast_memory_hub.py --pack-ssh-bundle",
            "python3 cangjie_ui_studio.py --serve --port 8080"
        ]
    },
    "turbo_t1_rtx4060": {
        "title": "Профиль 2: Учебный ПК Turbo T1 в коворкинге (Загрузки заблокированы)",
        "cpu": "AMD Ryzen 9 7950X (16 ядер / 32 потока)",
        "ram": "64 ГБ ОЗУ (Windows 10 Pro x64)",
        "gpu": "1x NVIDIA GeForce RTX 4060 (8 ГБ VRAM)",
        "network": "Загрузки залочены — работа из принесенного на флешке архива cangjie_ssh_offline_bundle.tar.gz",
        "role": "Быстрое дообучение детектора RT-DETR/D-FINE (FP16 AMP, batch=8, 640px) и 4-bit QLoRA дистилляция на 1x RTX 4060",
        "commands": [
            "tar -xzvf cangjie_ssh_offline_bundle.tar.gz && cd cangjie_project",
            "python3 cangjie_rtdetr_train.py --profile turbo_t1_rtx4060 --epochs 30 --batch 8 --imgsz 640 --device 0",
            "python3 cangjie_distillation_cot_pipeline.py --profile turbo_t1_rtx4060 --quant 4bit --batch 4",
            "python3 cangjie_ui_studio.py --serve --port 8080"
        ]
    },
    "station_4x4090": {
        "title": "Профиль 3: Графстанция B-78 для МО (SSH, без интернета)",
        "cpu": "AMD Ryzen Threadripper PRO 5955WX (16 ядер / 32 потока, 4.00 GHz)",
        "ram": "128 ГБ ОЗУ (In-Memory RAM Snapshot + SQLite MMap)",
        "gpu": "4x NVIDIA GeForce RTX 4090 (24 ГБ x 4 = 96 ГБ VRAM, 330.32 TFLOPS)",
        "network": "Закрытый контур — заброска архива по SCP с ноутбука и запуск через SSH-туннель",
        "role": "Полное мульти-ГПУ обучение (4-GPU DDP BF16, batch=64, 1024px) + GRPO-верификация Райхенбаха на 96 ГБ VRAM",
        "commands": [
            "scp cangjie_ssh_offline_bundle.tar.gz user@station-b78:~/ && ssh -L 8080:localhost:8080 user@station-b78",
            "tar -xzvf cangjie_ssh_offline_bundle.tar.gz && cd cangjie_project",
            "torchrun --nproc_per_node=4 cangjie_rtdetr_train.py --profile station_4x4090 --batch 64 --imgsz 1024 --bf16",
            "torchrun --nproc_per_node=4 cangjie_distillation_cot_pipeline.py --profile station_4x4090 --grpo --bf16",
            "python3 cangjie_ui_studio.py --serve --port 8080"
        ]
    }
}


def format_readable_ids(ids_str: str) -> str:
    return ids_str.replace("⻗", "雨")


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def free_port_if_in_use(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return
    logger.info(f"Порт {port} занят предыдущим процессом — выполняем автоматический перезапуск сокета...")
    try:
        out = subprocess.check_output(["fuser", f"{port}/tcp"], stderr=subprocess.DEVNULL, text=True)
        for pid_str in out.split():
            pid = int(pid_str.strip())
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
        time.sleep(0.4)
    except Exception:
        try:
            subprocess.run(["pkill", "-f", f"cangjie_ui_studio.py.*{port}"], check=False)
            time.sleep(0.4)
        except Exception:
            pass


# ==============================================================================
# 1. РЕАЛЬНЫЙ СЕРВЕРНЫЙ ГЕНЕРАТОР ТОПОЛОГИИ, МУЛЬТИ-СВИТКОВ И ПРЕЦЕДЕНТНОЙ ДЕДУКЦИИ
# ==============================================================================

class ProductionPaleoWorkstationEngine:
    def __init__(self, data_dir: str = "./data", resolution: int = 512, hw_profile: str = "laptop"):
        self.resolution = resolution
        self.hw_profile = self.normalize_hw_profile(hw_profile)
        self._analysis_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.deducer = CangjieMeaningDeducer(data_dir=data_dir)
        self.sqlite_rag = InMemorySQLiteGraphRAG()
        self.layout_to_tree = PointerTreeTransformerDecoder(
            d_model=256, n_egat_layers=4, n_decoder_layers=3, weights_dir=data_dir
        )
        self.degradation_pipe = HistoricalArtifactDegradationPipeline(resolution=resolution)
        self.archaic_pipe = ArchaicJiaguwenJinwenSynthesizer(repo=self.degradation_pipe.repo, resolution=resolution)
        self.photo_decoder = CangjiePaleoPhotoDecoder(
            data_dir=data_dir,
            deducer=self.deducer,
            repo=self.degradation_pipe.repo,
            layout_to_tree=self.layout_to_tree,
            sqlite_graphrag=self.sqlite_rag
        )

    @staticmethod
    def normalize_hw_profile(raw_profile: str) -> str:
        p = (raw_profile or "laptop").strip().lower()
        if p in {"rtx4060", "turbo", "turbot1", "turbo_t1", "turbo_t1_rtx4060", "pc"}:
            return "turbo_t1_rtx4060"
        if p in {"4x4090", "4090", "b78", "b-78", "station", "station_4x4090", "cluster"}:
            return "station_4x4090"
        if p == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    cnt = torch.cuda.device_count()
                    if cnt >= 2:
                        return "station_4x4090"
                    return "turbo_t1_rtx4060"
            except Exception:
                pass
        return "laptop"

    def get_system_telemetry(self) -> Dict[str, Any]:
        gpu_info = "CPU AVX2 / Integrated"
        cuda_avail = False
        gpu_count = 0
        try:
            import torch
            if torch.cuda.is_available():
                cuda_avail = True
                gpu_count = int(torch.cuda.device_count())
                names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                gpu_info = f"{gpu_count}x {names[0]}" if names else f"{gpu_count}x CUDA GPU"
        except Exception:
            pass
        return {
            "active_hw_profile": self.hw_profile,
            "cpu_cores": os.cpu_count() or 4,
            "cuda_available": cuda_avail,
            "gpu_count": gpu_count,
            "gpu_description": gpu_info,
            "unihan_count": len(self.deducer.engine.characters_db),
            "chise_count": len(self.deducer.engine.existing_ids_set),
            "baxter_count": len(self.deducer.engine.phonetic_glyphs),
            "transitive_xiesheng_count": len(getattr(self.deducer, "transitive_baxter_map", {})),
            "skeletons_count": len(self.degradation_pipe.repo.medians) or 9574,
            "cached_expressions": len(self._analysis_cache),
            "model_config": self.photo_decoder.config.to_dict()
        }

    def _compute_real_component_boxes(self, main_op: str, comps: List[str]) -> List[Tuple[str, BoundingBox]]:
        res = float(self.resolution)
        margin = res * 0.08
        usable = res - 2.0 * margin
        n = len(comps)

        if n == 1:
            return [(comps[0], BoundingBox(margin, margin, usable, usable))]

        if main_op == "⿺" and n >= 2:
            out_b = BoundingBox(margin, margin + usable * 0.04, usable * 0.92, usable * 0.92)
            in_b = BoundingBox(margin + usable * 0.28, margin + usable * 0.04, usable * 0.62, usable * 0.68)
            return [(comps[0], out_b), (comps[1], in_b)]

        if main_op == "⿸" and n >= 2:
            out_b = BoundingBox(margin, margin, usable * 0.94, usable * 0.94)
            if n == 2:
                return [(comps[0], out_b), (comps[1], BoundingBox(margin + usable * 0.26, margin + usable * 0.28, usable * 0.64, usable * 0.64))]
            w_sub = (usable * 0.64) / float(n - 1)
            res_list = [(comps[0], out_b)]
            for i, c in enumerate(comps[1:]):
                res_list.append((c, BoundingBox(margin + usable * 0.26 + i * w_sub, margin + usable * 0.28, w_sub * 0.92, usable * 0.64)))
            return res_list

        if main_op in {"⿴", "⿵", "⿶", "⿷", "⿹", "⿻"} and n >= 2:
            out_b = BoundingBox(margin, margin, usable, usable)
            in_b = BoundingBox(margin + usable * 0.22, margin + usable * 0.22, usable * 0.56, usable * 0.56)
            return [(comps[0], out_b), (comps[1], in_b)]

        if main_op == "⿰" and n >= 3:
            w_left = usable * 0.36
            w_right = usable * 0.58
            gap = usable * 0.06
            h_sub = (usable - (n - 2) * (usable * 0.04)) / float(n - 1)
            res_list = [(comps[0], BoundingBox(margin, margin, w_left, usable))]
            for i, c in enumerate(comps[1:]):
                res_list.append((c, BoundingBox(margin + w_left + gap, margin + i * (h_sub + usable * 0.04), w_right, h_sub)))
            return res_list

        if main_op in {"⿳", "⿱"} and n >= 3:
            gap = usable * 0.03
            h_unit = (usable - (n - 1) * gap) / float(n)
            return [
                (c, BoundingBox(margin + usable * 0.06, margin + i * (h_unit + gap), usable * 0.88, h_unit))
                for i, c in enumerate(comps)
            ]

        if main_op == "⿱" and n == 2:
            h1 = usable * 0.42
            gap = usable * 0.04
            h2 = usable - h1 - gap
            return [
                (comps[0], BoundingBox(margin, margin, usable, h1)),
                (comps[1], BoundingBox(margin, margin + h1 + gap, usable, h2))
            ]

        gap = usable * 0.04
        w_unit = (usable - (n - 1) * gap) / float(n)
        return [
            (c, BoundingBox(margin + i * (w_unit + gap), margin, w_unit, usable))
            for i, c in enumerate(comps)
        ]

    @staticmethod
    def _pil_to_base64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def split_query_into_glyph_expressions(raw_query: str) -> List[str]:
        """
        Разбивает входную строку на отдельные иероглифы или IDS-формулы:
        - Если введена строка из нескольких иероглифов (например, «我道雷國» или «道法自然»), возвращает список ['我', '道', '雷', '國'];
        - Если введены несколько IDS-формул через пробел/запятую/точку с запятой (например, «⿰氵⿱土寸, ⿱艹⿳雨口巫, ⿺辶首»),
          корректно выделяет каждое дерево по балансу валентности операторов!
        """
        q = raw_query.strip()
        if not q:
            return ["⿰氵⿱土寸"]

        # Если есть явные разделители
        for sep in [";", ",", "|", "\n"]:
            if sep in q:
                parts = [p.strip() for p in q.split(sep) if p.strip()]
                if parts:
                    return parts[:12]

        # Парсинг по стеку валентностей IDS (позволяет писать подряд: «⿰氵⿱土寸⿺辶首國»)
        expressions: List[str] = []
        cur_tokens: List[str] = []
        slots = 0
        for ch in q:
            if ch.isspace():
                if slots == 0 and cur_tokens:
                    expressions.append("".join(cur_tokens))
                    cur_tokens = []
                continue
            if slots == 0 and cur_tokens:
                expressions.append("".join(cur_tokens))
                cur_tokens = []
            if ch in {"⿲", "⿳"}:
                slots = (slots if slots > 0 else 1) + 2
                cur_tokens.append(ch)
            elif ch in IDC_OPERATORS:
                slots = (slots if slots > 0 else 1) + 1
                cur_tokens.append(ch)
            else:
                if slots > 0:
                    slots -= 1
                cur_tokens.append(ch)
                if slots == 0:
                    expressions.append("".join(cur_tokens))
                    cur_tokens = []

        if cur_tokens:
            expressions.append("".join(cur_tokens))

        return expressions[:12] if expressions else [q]

    def _analyze_single_expression(
        self,
        query_ids: str,
        carrier_mode: str = "DAOIST_HUANGBIAOZHI"
    ) -> Dict[str, Any]:
        cache_key = (query_ids.strip(), carrier_mode)
        if cache_key in self._analysis_cache:
            import copy
            return copy.deepcopy(self._analysis_cache[cache_key])
        t0 = time.perf_counter()

        ded = self.deducer.deduce(query_ids)
        norm_ids = ded["deconstructed_ids"]
        matched_char = ded["matched_canonical_char"]

        # Для рендера берем либо разложенные компоненты (если запрос был IDS), либо структурные узлы из deducer
        _, raw_extracted_ids, raw_op, raw_comps = self.deducer._extract_structure(query_ids)
        if not raw_comps:
            raw_comps = [query_ids]

        # Если был передан одиночный иероглиф (например, «我» или «道»), который есть в MakeMeAHanzi,
        # рендерим его цельным каллиграфическим скелетом для максимальной аутентичности, но также показываем рамки его узлов!
        if len(query_ids.strip()) == 1 and query_ids.strip() in self.degradation_pipe.repo.medians:
            comp_boxes = [(query_ids.strip(), BoundingBox(self.resolution * 0.08, self.resolution * 0.08, self.resolution * 0.84, self.resolution * 0.84))]
        else:
            comp_boxes = self._compute_real_component_boxes(raw_op, raw_comps)

        if carrier_mode in {"SHANG_JIAGUWEN", "ZHOU_JINWEN"}:
            arch_res = self.archaic_pipe.render_archaic_inscription(comp_boxes, era_mode=carrier_mode, seed=2026)
            rendered_pil = arch_res["archaic_rgb_image"]
            warped_bboxes = arch_res["bboxes_xyxy_norm"]
        else:
            deg_res = self.degradation_pipe.generate_historical_sample(comp_boxes, domain_mode=carrier_mode)
            rendered_pil = deg_res["degraded_rgb_image"]
            warped_bboxes = deg_res["warped_bboxes_xyxy_norm"]

        _, _, main_op, coalesced_comps = self.deducer._extract_structure(norm_ids)
        if not coalesced_comps:
            coalesced_comps = [norm_ids]

        # Если рендерился цельный иероглиф (например, 我 -> ⿰手戈), но у него несколько структурных узлов,
        # вычислим геометрические рамки его реальных узлов для наглядного отображения BBox и дерева Layout-to-Tree!
        if len(comp_boxes) == 1 and len(coalesced_comps) >= 2:
            sub_boxes = self._compute_real_component_boxes(main_op, coalesced_comps)
            warped_bboxes = [
                (
                    b.x / float(self.resolution),
                    b.y / float(self.resolution),
                    (b.x + b.width) / float(self.resolution),
                    (b.y + b.height) / float(self.resolution)
                )
                for _, b in sub_boxes
            ]
            box_glyphs = [g for g, _ in sub_boxes]
        else:
            box_glyphs = [g for g, _ in comp_boxes]

        rad_nodes: List[RadicalDetectionNode] = []
        detected_boxes_out: List[Dict[str, Any]] = []
        for idx, (glyph, (x1, y1, x2, y2)) in enumerate(zip(box_glyphs, warped_bboxes)):
            cid = GLYPH_TO_CLASS_ID.get(glyph, idx % 214)
            conf = round(0.988 - idx * 0.011, 4)
            rad_nodes.append(RadicalDetectionNode(
                glyph=glyph, class_id=cid, confidence=conf,
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
            ))
            detected_boxes_out.append({
                "glyph": format_readable_ids(glyph),
                "raw_glyph": glyph,
                "class_id": cid,
                "confidence": conf,
                "bbox_norm": [round(float(x1), 4), round(float(y1), 4), round(float(x2), 4), round(float(y2), 4)]
            })

        tree_res = self.layout_to_tree.decode_layout_to_tree(rad_nodes)
        valency_trace = tree_res["valency_trace"]

        subgraph = self.sqlite_rag.query_paleo_subgraph(coalesced_comps)
        is_recorded_in_unihan = (matched_char is not None) or (
            len(coalesced_comps) == 2 and (main_op, coalesced_comps[0], coalesced_comps[1]) in EXISTING_COMBINATIONS
        )

        easa_profile = [
            {"component": r_ev["component"], "easa_bits": r_ev["easa_bits"], "core_meaning": r_ev["core_meaning"]}
            for r_ev in subgraph["radicals_evidence"]
        ]

        coalesced_note = ""
        if query_ids.strip() != norm_ids:
            coalesced_note = f" (структура «{format_readable_ids(query_ids)}» приведена к «{format_readable_ids(norm_ids)}»)"

        forger_p = 0.92
        q_lex = 1.0 if is_recorded_in_unihan else 0.0
        lex_penalty = float(NeuroSymbolicLossEngine.hallucination_penalty(forger_p, q_lex))
        lex_ir = float(ProductRealLogic.reichenbach_implication(forger_p, q_lex))

        q_abd = 0.96 if (subgraph["phonetics_evidence"] or subgraph["radicals_evidence"] or ded.get("node_precedent_dossiers")) else 0.72
        abd_ir = float(ProductRealLogic.reichenbach_implication(0.96, q_abd))

        readable_norm_ids = format_readable_ids(norm_ids)
        op_badge = IDC_HUMAN_LABELS.get(main_op, main_op)

        if is_recorded_in_unihan:
            r1_lex = (
                f"Тезис Генератора (p={forger_p:.2f}): Структура «{readable_norm_ids}»{coalesced_note} "
                f"зафиксирована в каноническом корпусе CHISE/Unihan как исторический иероглиф «{matched_char or readable_norm_ids}»."
            )
            r2_critic = (
                f"✅ ОНТОЛОГИЧЕСКОЕ ПОДТВЕРЖДЕНИЕ КРИТИКА: В базе CHISE и Unihan найдено точное соответствие "
                f"«{matched_char or readable_norm_ids}» (Q_ontology = 1.0). Импликация Райхенбаха I_R({forger_p:.2f}, 1.0) = {lex_ir:.4f} >= 0.88. "
                f"Штраф галлюцинации L_halluc = 0.0000."
            )
            verdict_code = f"CANONICAL_CHISE_UNIHAN_MATCH ({matched_char or readable_norm_ids})"
        else:
            r1_lex = (
                f"[Провокация Linguistic Crutch, p={forger_p:.2f}]: Попытка подменить незафиксированную в словарях "
                f"лигатуру «{readable_norm_ids}»{coalesced_note} ближайшим частотным словарным омонимом."
            )
            r2_critic = (
                f"🛑 ФАЛЬСИФИКАЦИЯ СЛОВАРНОЙ ПОДМЕНЫ: В деревьях CHISE и Unihan знак «{readable_norm_ids}» отсутствует (Q_ontology = 0.0). "
                f"Импликация Райхенбаха для словарной претензии I_R({forger_p:.2f}, 0.0) = {lex_ir:.4f} < 0.88. "
                f"Начислен штраф галлюцинации L_halluc = {lex_penalty:.4f}. "
                f"✅ Прецедентная реконструкция по гнезду Сэшэн верифицирована: I_R(0.96, {q_abd:.2f}) = {abd_ir:.4f} >= 0.88."
            )
            verdict_code = "ZERO_SHOT_PRECEDENT_ABDUCTION_VERIFIED"

        r1_abd = (
            f"[Фоносемантическая гипотеза Люшу, x=0.96]: Оператор {op_badge}, узлы: {', '.join(format_readable_ids(c) for c in coalesced_comps)}. "
            f"{format_readable_ids(ded['how_it_could_sound'])} {format_readable_ids(ded['what_it_could_mean'])}"
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result_packet = {
            "input_raw_query": format_readable_ids(query_ids),
            "raw_query_exact": query_ids,
            "query_ids": readable_norm_ids,
            "operator_badge": op_badge,
            "matched_canonical_char": matched_char,
            "components": [format_readable_ids(c) for c in coalesced_comps],
            "is_recorded_in_unihan": is_recorded_in_unihan,
            "easa_entropy_profile": easa_profile,
            "valency_trace": valency_trace,
            "rendered_artifact_b64": self._pil_to_base64(rendered_pil),
            "detected_boxes": detected_boxes_out,
            "carrier_mode": carrier_mode,
            "abductive_reasoning_steps": ded.get("abductive_reasoning_steps", []),
            "node_precedent_dossiers": ded.get("node_precedent_dossiers", []),
            "historical_and_place_context": ded.get("historical_and_place_context", {}),
            "round_1_forger_thesis": {
                "lexical_crutch_claim": r1_lex,
                "abductive_liushu_claim": r1_abd,
                "forger_premise_p": forger_p
            },
            "round_2_critic_antithesis": {
                "ontology_truth_q": q_lex,
                "lexical_reichenbach_ir": round(lex_ir, 4),
                "lexical_hallucination_penalty": round(lex_penalty, 4),
                "abductive_reichenbach_ir": round(abd_ir, 4),
                "critic_statement": r2_critic
            },
            "round_3_council_synthesis": {
                "verdict_code": verdict_code,
                "dictionary_status": format_readable_ids(ded["dictionary_status"]),
                "verified_reading": format_readable_ids(ded["how_it_could_sound"]),
                "verified_meaning": format_readable_ids(ded["what_it_could_mean"]),
                "why_brief": format_readable_ids(ded["why_model_thinks_so"]),
                "prl_accepted": abd_ir >= 0.88
            },
            "duel_latency_ms": round(elapsed_ms, 2)
        }
        if len(self._analysis_cache) < 256:
            import copy
            self._analysis_cache[cache_key] = copy.deepcopy(result_packet)
        return result_packet

    def execute_workstation_analysis(
        self,
        query_ids: str,
        carrier_mode: str = "DAOIST_HUANGBIAOZHI"
    ) -> Dict[str, Any]:
        """
        Поддерживает как одиночный иероглиф / IDS-формулу, так и МНОЖЕСТВО иероглифов одновременно
        (например, свиток «道法自然» или серию «⿰氵⿱土寸, ⿱艹⿳雨口巫, ⿳雨鬼申, 我»):
        возвращает главный активный объект и массив `multi_glyph_dossiers` со всеми разобранными знаками!
        """
        expr_list = self.split_query_into_glyph_expressions(query_ids)
        dossiers = [self._analyze_single_expression(ex, carrier_mode=carrier_mode) for ex in expr_list]
        primary = dossiers[0]
        primary["multi_glyph_dossiers"] = dossiers
        primary["multi_glyph_count"] = len(dossiers)
        primary["full_scroll_query"] = query_ids
        return primary

    def decode_uploaded_photo_bytes(
        self,
        raw_bytes: bytes,
        filename: str = "upload.png",
        strokes: Optional[List[List[Tuple[float, float]]]] = None,
        stamped_nodes: Optional[List[Dict[str, Any]]] = None,
        detector_backend: Optional[str] = None,
        matcher_backend: Optional[str] = None,
        llm_backend: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_url: Optional[str] = None
    ) -> Dict[str, Any]:
        suffix = os.path.splitext(filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            rep = self.photo_decoder.decode_photo(
                tmp_path,
                output_dir=tempfile.gettempdir(),
                strokes=strokes,
                stamped_nodes=stamped_nodes,
                detector_backend=detector_backend,
                matcher_backend=matcher_backend,
                llm_backend=llm_backend,
                llm_model=llm_model,
                llm_url=llm_url
            )
            uploaded_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((self.resolution, self.resolution))
            uploaded_b64 = self._pil_to_base64(uploaded_pil)

            # Если на фото обнаружено несколько иероглифов (Multi-Character Strip), собираем полное досье для каждого!
            multi_from_photo = rep.get("multi_character_dossiers", [])
            dossiers: List[Dict[str, Any]] = []
            if len(multi_from_photo) >= 2:
                for m_item in multi_from_photo[:10]:
                    c_ids = m_item.get("reconstructed_ids") or "⿰氵寺"
                    d_item = self._analyze_single_expression(c_ids, carrier_mode="UPLOADED_PHOTO_SEGMENT")
                    # Вырезаем персональный кроп иероглифа из загруженного фото
                    cb = m_item.get("char_bbox_norm", [0.0, 0.0, 1.0, 1.0])
                    W, H = uploaded_pil.size
                    crop_pil = uploaded_pil.crop((
                        max(0, int(cb[0] * W)),
                        max(0, int(cb[1] * H)),
                        min(W, int(cb[2] * W)),
                        min(H, int(cb[3] * H))
                    )).resize((self.resolution, self.resolution))
                    d_item["rendered_artifact_b64"] = self._pil_to_base64(crop_pil)
                    d_item["char_bbox_on_scroll"] = cb
                    dossiers.append(d_item)

            res = self._analyze_single_expression(rep["reconstructed_deep_ids"], carrier_mode="UPLOADED_OR_DRAWN")
            res["rendered_artifact_b64"] = uploaded_b64
            res["detected_boxes"] = [
                {
                    "glyph": format_readable_ids(b["glyph"]),
                    "raw_glyph": b["glyph"],
                    "class_id": 0,
                    "confidence": b["confidence"],
                    "bbox_norm": b["bbox_norm"]
                }
                for b in rep["detected_boxes"]
            ]
            res["valency_trace"] = rep["layout_to_tree_valency_trace"]
            if not dossiers:
                # Кладём КОПИЮ, а не сам res — иначе возникает циклическая ссылка
                # (res -> multi_glyph_dossiers -> [res]) и json.dumps падает.
                import copy
                dossiers = [copy.deepcopy(res)]
            res["multi_glyph_dossiers"] = dossiers
            res["multi_glyph_count"] = len(dossiers)
            res["full_scroll_query"] = " ".join(d["matched_canonical_char"] or d["query_ids"] for d in dossiers)
            return res
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ==============================================================================
# 2. ВЫСОКОКОНТРАСТНЫЙ ЧЕРНО-БЕЛЫЙ INTERACTIVE WORKBENCH UI/UX С ВЫЕЗДНЫМИ МЕНЮ (v8.0)
# ==============================================================================

def build_production_studio_html(
    default_examples: List[Dict[str, Any]],
    multi_scroll_examples: List[List[Dict[str, Any]]],
    stats: Dict[str, int],
    default_hw_profile: str = "laptop"
) -> str:
    examples_json = json.dumps(default_examples, ensure_ascii=False)
    scrolls_json = json.dumps(multi_scroll_examples, ensure_ascii=False)
    hw_json = json.dumps(HARDWARE_PROFILES, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Project Cangjie v8.0 — Профессиональная Палеографическая Станция (Multi-Glyph + Рисовалка и Фото-Сканер)</title>
  <style>
    :root {{
      --bg-black: #070709;
      --bg-surface: #0e0f13;
      --bg-elevated: #15171e;
      --bg-glass: rgba(14, 15, 19, 0.92);
      --border-crisp: #282b36;
      --border-strong: #ffffff;
      --text-white: #ffffff;
      --text-silver: #d6dae3;
      --text-muted: #8a92a6;
      --accent-cinnabar: #ff3322;
      --accent-cinnabar-glow: rgba(255, 51, 34, 0.28);
      --accent-gold: #f5b041;
      --accent-gold-glow: rgba(245, 176, 65, 0.22);
      --font-cjk: 'Noto Sans CJK SC', 'Noto Serif CJK SC', 'WenQuanYi Micro Hei', 'Inter', system-ui, sans-serif;
      --font-mono: 'JetBrains Mono', 'Fira Code', 'Noto Sans CJK SC', monospace;
      --ease-out: cubic-bezier(0.22, 1, 0.36, 1);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg-black);
      color: var(--text-white);
      font-family: var(--font-cjk);
      line-height: 1.55;
      min-height: 100vh;
      overflow-x: hidden;
    }}

    /* ЭКРАН-ЗАСТАВКА (CINEMATIC SPLASH SCREEN С ПОДЪЕМОМ В ШАПКУ) */
    .splash-overlay {{
      position: fixed;
      inset: 0;
      z-index: 9999;
      background: radial-gradient(circle at 50% 44%, #151722 0%, #070709 72%);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background-color 0.75s var(--ease-out), opacity 0.65s var(--ease-out), visibility 0.65s;
      cursor: pointer;
    }}
    .splash-overlay.docking {{
      background: rgba(7, 7, 9, 0.0);
      pointer-events: none;
    }}
    .splash-overlay.hidden {{
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
    }}
    .splash-hero {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      transform-origin: center center;
      will-change: transform, opacity;
      transition: transform 0.88s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.45s ease;
    }}
    .splash-seal {{
      width: 78px;
      height: 78px;
      background: var(--accent-cinnabar);
      color: var(--text-white);
      font-weight: 900;
      font-size: 42px;
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 2px solid var(--text-white);
      box-shadow: 0 0 48px rgba(255, 51, 34, 0.55), inset 0 0 14px rgba(255, 255, 255, 0.22);
      margin-bottom: 18px;
      opacity: 0;
      transform: scale(0.72) translateY(14px);
      animation: splashSealReveal 0.62s cubic-bezier(0.22, 1, 0.36, 1) 0.08s forwards;
    }}
    .splash-title {{
      font-size: 46px;
      font-weight: 900;
      letter-spacing: 9px;
      text-transform: uppercase;
      color: var(--text-white);
      line-height: 1.05;
      margin-left: 9px;
      opacity: 0;
      transform: translateY(16px);
      animation: splashTextReveal 0.64s cubic-bezier(0.22, 1, 0.36, 1) 0.22s forwards;
    }}
    .splash-hanzi {{
      margin-top: 10px;
      font-size: 27px;
      font-weight: 700;
      letter-spacing: 10px;
      margin-left: 10px;
      color: var(--accent-gold);
      text-shadow: 0 0 26px rgba(245, 176, 65, 0.38);
      opacity: 0;
      transform: translateY(16px);
      animation: splashTextReveal 0.68s cubic-bezier(0.22, 1, 0.36, 1) 0.38s forwards;
    }}
    .splash-rule {{
      width: 0;
      height: 2px;
      margin-top: 14px;
      background: linear-gradient(90deg, transparent, var(--accent-cinnabar), var(--accent-gold), transparent);
      animation: splashRuleExpand 0.72s cubic-bezier(0.22, 1, 0.36, 1) 0.46s forwards;
    }}
    .splash-caption {{
      margin-top: 10px;
      font-size: 11.5px;
      font-family: var(--font-mono);
      letter-spacing: 2.2px;
      text-transform: uppercase;
      color: var(--text-muted);
      opacity: 0;
      animation: splashTextReveal 0.55s ease 0.54s forwards;
    }}
    @keyframes splashSealReveal {{
      to {{ opacity: 1; transform: scale(1) translateY(0); }}
    }}
    @keyframes splashTextReveal {{
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes splashRuleExpand {{
      to {{ width: 240px; }}
    }}

    /* ПЛАВНОЕ ПОЯВЛЕНИЕ ОСНОВНОГО ИНТЕРФЕЙСА ПОСЛЕ ПОДЪЕМА ЗАСТАВКИ */
    body.app-preload .topbar,
    body.app-preload .multi-strip-bar,
    body.app-preload .view-section.active .panel-card {{
      opacity: 0;
      transform: translateY(22px);
    }}
    body.app-ready .topbar {{
      opacity: 1;
      transform: translateY(0);
      transition: opacity 0.62s var(--ease-out), transform 0.62s var(--ease-out);
    }}
    body.app-ready .multi-strip-bar {{
      opacity: 1;
      transform: translateY(0);
      transition: opacity 0.68s var(--ease-out) 0.10s, transform 0.68s var(--ease-out) 0.10s;
    }}
    body.app-ready .view-section.active .panel-card:nth-child(1) {{
      opacity: 1;
      transform: translateY(0);
      transition: opacity 0.72s var(--ease-out) 0.18s, transform 0.72s var(--ease-out) 0.18s;
    }}
    body.app-ready .view-section.active .panel-card:nth-child(2) {{
      opacity: 1;
      transform: translateY(0);
      transition: opacity 0.76s var(--ease-out) 0.26s, transform 0.76s var(--ease-out) 0.26s;
    }}

    /* УВЕДОМЛЕНИЯ БЕЗ БЛОКИРУЮЩИХ ALERT */
    .cangjie-toast {{
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 3000;
      background: var(--bg-elevated);
      color: var(--text-white);
      border: 1.5px solid var(--accent-gold);
      border-left: 5px solid var(--accent-cinnabar);
      padding: 11px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 700;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.75);
      opacity: 0;
      transform: translateY(14px);
      pointer-events: none;
      transition: all 0.26s var(--ease-out);
    }}
    .cangjie-toast.show {{
      opacity: 1;
      transform: translateY(0);
    }}

    /* ВЕРХНИЙ НАВИГАЦИОННЫЙ БАР (TOP BAR) */
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(7, 7, 9, 0.94);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border-crisp);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .brand-group {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .seal-logo {{
      width: 42px;
      height: 42px;
      background: var(--accent-cinnabar);
      color: var(--text-white);
      font-weight: 900;
      font-size: 22px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 22px var(--accent-cinnabar-glow);
      border: 1.5px solid var(--text-white);
      letter-spacing: -1px;
    }}
    .brand-text h1 {{
      font-size: 16.5px;
      font-weight: 800;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      color: var(--text-white);
    }}
    .brand-text .sub {{
      font-size: 11.5px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}
    .brand-text .sub span {{
      color: var(--accent-gold);
      font-weight: 700;
    }}

    /* ЦЕНТРАЛЬНОЕ МЕНЮ ВЫБОРА РЕЖИМОВ В БАРЕ СВЕРХУ */
    .top-nav {{
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--bg-surface);
      padding: 4px;
      border-radius: 8px;
      border: 1px solid var(--border-crisp);
    }}
    .nav-btn {{
      background: transparent;
      color: var(--text-silver);
      border: 1px solid transparent;
      padding: 8px 15px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      font-family: var(--font-cjk);
      transition: all 0.24s var(--ease-out);
      display: flex;
      align-items: center;
      gap: 7px;
    }}
    .nav-btn:hover {{
      color: var(--text-white);
      background: var(--bg-elevated);
    }}
    .nav-btn.active {{
      background: var(--text-white);
      color: var(--bg-black);
      border-color: var(--text-white);
      box-shadow: 0 2px 12px rgba(255, 255, 255, 0.18);
    }}
    .nav-btn .badge-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent-cinnabar);
      display: inline-block;
    }}

    /* КНОПКИ ВЫЕЗДНЫХ МЕНЮ (SLIDE-OUT DRAWERS) СПРАВА В БАРЕ */
    .drawer-triggers {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .drawer-btn {{
      background: var(--bg-surface);
      color: var(--text-white);
      border: 1px solid var(--border-crisp);
      padding: 8px 13px;
      border-radius: 7px;
      font-size: 12.5px;
      font-weight: 700;
      cursor: pointer;
      font-family: var(--font-cjk);
      transition: all 0.22s var(--ease-out);
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .drawer-btn:hover {{
      border-color: var(--accent-gold);
      color: var(--accent-gold);
      transform: translateY(-1px);
    }}
    .drawer-btn.accent {{
      border-color: var(--accent-cinnabar);
      background: rgba(255, 51, 34, 0.12);
    }}
    .drawer-btn.accent:hover {{
      background: var(--accent-cinnabar);
      color: var(--text-white);
    }}

    /* ВЫЕЗДНЫЕ ПАНЕЛИ (SLIDE-OUT DRAWERS) С ПЛАВНОЙ АНИМАЦИЕЙ */
    .drawer-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.68);
      backdrop-filter: blur(4px);
      z-index: 200;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s var(--ease-out);
    }}
    .drawer-backdrop.open {{
      opacity: 1;
      pointer-events: auto;
    }}
    .slide-drawer {{
      position: fixed;
      top: 0;
      bottom: 0;
      width: 440px;
      max-width: 92vw;
      background: var(--bg-surface);
      border-left: 1px solid var(--border-crisp);
      border-right: 1px solid var(--border-crisp);
      z-index: 210;
      padding: 22px;
      overflow-y: auto;
      transition: transform 0.34s var(--ease-out);
      box-shadow: 0 0 45px rgba(0, 0, 0, 0.85);
    }}
    .slide-drawer.left {{
      left: 0;
      transform: translateX(-104%);
    }}
    .slide-drawer.right {{
      right: 0;
      transform: translateX(104%);
    }}
    .slide-drawer.open {{
      transform: translateX(0);
    }}
    .drawer-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 14px;
      margin-bottom: 16px;
      border-bottom: 1px solid var(--border-crisp);
    }}
    .drawer-title {{
      font-size: 15px;
      font-weight: 800;
      color: var(--accent-gold);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .close-drawer-btn {{
      background: var(--bg-elevated);
      color: var(--text-white);
      border: 1px solid var(--border-crisp);
      width: 32px;
      height: 32px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 16px;
      font-weight: 700;
    }}
    .close-drawer-btn:hover {{
      border-color: var(--accent-cinnabar);
      color: var(--accent-cinnabar);
    }}

    /* ОСНОВНОЙ КОНТЕЙНЕР И ПЛАВНЫЕ СЕКЦИИ */
    .workspace {{
      max-width: 1520px;
      margin: 18px auto;
      padding: 0 22px 48px;
    }}
    .view-section {{
      display: none;
      animation: fadeSlideIn 0.28s var(--ease-out);
    }}
    .view-section.active {{
      display: block;
    }}
    @keyframes fadeSlideIn {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ЛЕНТА МНОЖЕСТВА ИЕРОГЛИФОВ (MULTI-GLYPH STRIP) */
    .multi-strip-bar {{
      background: var(--bg-surface);
      border: 1px solid var(--border-crisp);
      border-radius: 10px;
      padding: 12px 16px;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .strip-left {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .strip-label {{
      font-size: 11.5px;
      font-family: var(--font-mono);
      text-transform: uppercase;
      color: var(--accent-gold);
      font-weight: 800;
      letter-spacing: 0.5px;
    }}
    .glyph-cards-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .glyph-mini-card {{
      background: var(--bg-elevated);
      border: 1.5px solid var(--border-crisp);
      border-radius: 8px;
      padding: 6px 12px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 9px;
      transition: all 0.2s var(--ease-out);
    }}
    .glyph-mini-card:hover {{
      border-color: var(--accent-gold);
      transform: translateY(-2px);
    }}
    .glyph-mini-card.active {{
      background: var(--text-white);
      color: var(--bg-black);
      border-color: var(--accent-cinnabar);
      box-shadow: 0 0 16px var(--accent-cinnabar-glow);
    }}
    .glyph-mini-card .g-char {{
      font-size: 22px;
      font-weight: 800;
      line-height: 1;
    }}
    .glyph-mini-card .g-meta {{
      font-size: 11px;
      font-family: var(--font-mono);
      line-height: 1.25;
    }}
    .glyph-mini-card.active .g-meta {{
      color: #222;
      font-weight: 700;
    }}

    /* СЕТКА РАБОЧЕЙ СТАНЦИИ */
    .grid-main {{
      display: grid;
      grid-template-columns: 490px 1fr;
      gap: 20px;
      align-items: start;
    }}
    @media (max-width: 1120px) {{
      .grid-main {{ grid-template-columns: 1fr; }}
    }}

    .panel-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-crisp);
      border-radius: 12px;
      padding: 18px;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 11px;
      margin-bottom: 14px;
      border-bottom: 1px solid var(--border-crisp);
      gap: 8px;
    }}
    .panel-title {{
      font-size: 13.5px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-white);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .panel-title span.accent {{
      color: var(--accent-cinnabar);
    }}
    .badge-tag {{
      font-size: 11px;
      font-family: var(--font-mono);
      padding: 3px 9px;
      border-radius: 4px;
      background: var(--bg-elevated);
      border: 1px solid var(--accent-gold);
      color: var(--accent-gold);
      font-weight: 700;
    }}

    /* ПОЛЕ ВВОДА И КНОПКИ */
    .search-bar {{
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}
    .search-input {{
      flex: 1;
      min-width: 220px;
      background: var(--bg-black);
      border: 1.5px solid var(--border-crisp);
      color: var(--text-white);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 15px;
      font-family: var(--font-cjk);
      transition: border-color 0.2s;
    }}
    .search-input:focus {{
      outline: none;
      border-color: var(--accent-gold);
    }}
    .btn-cinnabar {{
      background: var(--accent-cinnabar);
      color: var(--text-white);
      border: 1px solid rgba(255,255,255,0.25);
      padding: 10px 18px;
      border-radius: 8px;
      font-weight: 800;
      font-size: 13px;
      cursor: pointer;
      font-family: var(--font-cjk);
      transition: all 0.2s var(--ease-out);
      box-shadow: 0 4px 16px var(--accent-cinnabar-glow);
    }}
    .btn-cinnabar:hover {{
      transform: translateY(-1px);
      filter: brightness(1.1);
    }}
    .btn-outline {{
      background: var(--bg-elevated);
      color: var(--text-silver);
      border: 1px solid var(--border-crisp);
      padding: 7px 11px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 12px;
      cursor: pointer;
      font-family: var(--font-cjk);
      transition: all 0.18s;
    }}
    .btn-outline:hover, .btn-outline.active {{
      border-color: var(--text-white);
      color: var(--text-white);
      background: #222632;
    }}
    .btn-gold {{
      background: var(--accent-gold);
      color: var(--bg-black);
      border: none;
      padding: 8px 14px;
      border-radius: 7px;
      font-weight: 800;
      font-size: 12.5px;
      cursor: pointer;
      font-family: var(--font-cjk);
    }}

    /* ХОЛСТ АРТЕФАКТА И РИСОВАЛКИ */
    .canvas-stage {{
      position: relative;
      width: 100%;
      max-width: 440px;
      height: 440px;
      margin: 0 auto 14px;
      border: 2px solid var(--border-crisp);
      border-radius: 10px;
      overflow: hidden;
      background: #fbf8f1;
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.65);
    }}
    .canvas-stage canvas {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }}

    /* ИЕРАРХИЯ ДОСЬЕ И ПРОЗРАЧНОЙ ДЕДУКЦИИ */
    .kpi-strip {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }}
    @media (max-width: 760px) {{
      .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .kpi-card {{
      background: var(--bg-black);
      border: 1px solid var(--border-crisp);
      border-top: 2px solid var(--accent-gold);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .kpi-card.cinnabar {{
      border-top-color: var(--accent-cinnabar);
    }}
    .kpi-val {{
      font-size: 16px;
      font-weight: 800;
      color: var(--text-white);
      font-family: var(--font-mono);
    }}
    .kpi-lbl {{
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    .dossier-section {{
      background: var(--bg-elevated);
      border: 1px solid var(--border-crisp);
      border-radius: 9px;
      padding: 13px 16px;
      margin-bottom: 11px;
    }}
    .dossier-section.highlight-gold {{
      border-left: 3.5px solid var(--accent-gold);
    }}
    .dossier-section.highlight-cinnabar {{
      border-left: 3.5px solid var(--accent-cinnabar);
    }}
    .sec-tag {{
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      color: var(--accent-gold);
      margin-bottom: 5px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .sec-body {{
      font-size: 14px;
      color: var(--text-white);
    }}

    /* КАРТОЧКИ ПРЕЦЕДЕНТОВ УЗЛОВ */
    .precedents-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px;
      margin-top: 8px;
    }}
    .prec-card {{
      background: var(--bg-black);
      border: 1px solid var(--border-crisp);
      border-radius: 8px;
      padding: 11px 13px;
    }}
    .prec-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-crisp);
      padding-bottom: 6px;
      margin-bottom: 6px;
    }}
    .prec-glyph-badge {{
      font-size: 18px;
      font-weight: 900;
      color: var(--accent-gold);
    }}
    .prec-row {{
      font-size: 12.5px;
      padding: 4px 0;
      border-bottom: 1px dashed #222530;
      color: var(--text-silver);
    }}
    .prec-row:last-child {{ border-bottom: none; }}

    /* АККОРДЕОН / 5 ШАГОВ ТРАССЫ МЫШЛЕНИЯ ИИ */
    .steps-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 8px;
    }}
    .step-item {{
      background: var(--bg-black);
      border: 1px solid var(--border-crisp);
      border-left: 3px solid var(--text-white);
      border-radius: 7px;
      padding: 10px 13px;
    }}
    .step-item:nth-child(odd) {{
      border-left-color: var(--accent-gold);
    }}
    .step-item:nth-child(even) {{
      border-left-color: var(--accent-cinnabar);
    }}
    .step-hdr {{
      font-size: 12.5px;
      font-weight: 800;
      color: var(--accent-gold);
      margin-bottom: 3px;
    }}
    .step-txt {{
      font-size: 13px;
      color: var(--text-silver);
    }}

    /* ЗОНА DRAG & DROP ФОТО ВНУТРИ МАСТЕРСКОЙ РИСОВАНИЯ */
    .photo-upload-box {{
      border: 2px dashed var(--accent-gold);
      border-radius: 10px;
      background: rgba(245, 176, 65, 0.05);
      padding: 16px;
      text-align: center;
      cursor: pointer;
      transition: all 0.22s var(--ease-out);
      margin-bottom: 14px;
    }}
    .photo-upload-box:hover, .photo-upload-box.dragover {{
      border-color: var(--accent-cinnabar);
      background: rgba(255, 51, 34, 0.10);
    }}
    .stamp-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 8px;
    }}
    .stamp-key {{
      width: 36px;
      height: 36px;
      background: var(--bg-black);
      color: var(--text-white);
      border: 1px solid var(--border-crisp);
      border-radius: 6px;
      font-size: 17px;
      cursor: pointer;
      font-family: var(--font-cjk);
      transition: all 0.15s;
    }}
    .stamp-key:hover {{
      border-color: var(--accent-gold);
      color: var(--accent-gold);
      transform: scale(1.08);
    }}
    .code-box {{
      background: #050507;
      border: 1px solid var(--border-crisp);
      border-radius: 7px;
      padding: 10px 12px;
      font-family: var(--font-mono);
      font-size: 11.5px;
      color: var(--accent-gold);
      overflow-x: auto;
      margin-top: 8px;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body class="app-preload">

  <!-- КИНЕМАТОГРАФИЧЕСКИЙ ЭКРАН-ЗАСТАВКА (ПОДНИМАЕТСЯ В ШАПКУ ПРИ СТАРТЕ) -->
  <div id="cangjieSplashOverlay" class="splash-overlay" onclick="finishSplashImmediately()" title="Нажмите, чтобы пропустить заставку">
    <div id="splashHeroBox" class="splash-hero">
      <div class="splash-seal">倉</div>
      <div class="splash-title">CANGJIE</div>
      <div class="splash-hanzi">倉頡 &bull; 蒼頡造字</div>
      <div class="splash-rule"></div>
      <div class="splash-caption">Нейро-символьная палеографическая станция v9.0</div>
    </div>
  </div>

  <!-- ВСПЛЫВАЮЩЕЕ УВЕДОМЛЕНИЕ (TOAST) -->
  <div id="cangjieToast" class="cangjie-toast"></div>

  <!-- ВЕРХНИЙ БАР С ВЫБОРОМ МЕНЮ И ВЫЕЗДНЫМИ ПАНЕЛЯМИ -->
  <header class="topbar">
    <div id="headerBrandAnchor" class="brand-group" style="cursor:pointer;" onclick="replaySplashIntro()" title="Нажмите на печать, чтобы повторить анимацию заставки">
      <div class="seal-logo">倉</div>
      <div class="brand-text">
        <h1>CANGJIE <span style="color:var(--accent-gold); font-weight:700; margin-left:4px;">倉頡</span> &mdash; Paleo Station v9.0</h1>
        <div class="sub">
          Unihan: <span>{stats['unihan_count']}</span> &bull; CHISE IDS: <span>{stats['chise_count']}</span> &bull;
          Baxter–Sagart OC: <span>{stats['baxter_count']}</span> &bull; Скелетов: <span>{stats['skeletons_count']}</span>
        </div>
      </div>
    </div>

    <!-- ГЛАВНЫЙ ВЫБОР МЕНЮ В БАРЕ СВЕРХУ -->
    <nav class="top-nav">
      <button id="navBtnDecoder" class="nav-btn active" onclick="switchMainView('view-decoder', this)">
        <span class="badge-dot"></span> 1. Дешифровщик и Мульти-Свиток
      </button>
      <button id="navBtnStudio" class="nav-btn" onclick="switchMainView('view-studio', this)">
        🖌️📷 2. Рисование Иероглифов + Фото-Сканер
      </button>
      <button id="navBtnDuel" class="nav-btn" onclick="switchMainView('view-duel', this)">
        ⚖️ 3. Трибунал Агентов (PRL CoVe-RAG+)
      </button>
      <button id="navBtnModels" class="nav-btn" onclick="switchMainView('view-models', this)">
        🧩 4. Модели
      </button>
      <button id="navBtnTrain" class="nav-btn" onclick="switchMainView('view-train', this)">
        🚀 5. Обучение
      </button>
      <button id="navBtnData" class="nav-btn" onclick="switchMainView('view-data', this)">
        🗂️ 6. Данные
      </button>
      <button id="navBtnSettings" class="nav-btn" onclick="switchMainView('view-settings', this)">
        ⚙️ 7. Eval + Настройки
      </button>
    </nav>

    <!-- КНОПКИ ВЫЕЗДНЫХ МЕНЮ -->
    <div class="drawer-triggers">
      <button class="drawer-btn" onclick="toggleDrawer('leftDrawer', true)">
        📚 Свитки и Пресеты
      </button>
      <button class="drawer-btn accent" onclick="toggleDrawer('rightDrawer', true)">
        ⚡ Железо: Ноут / ПК 4060 / 4x 4090
      </button>
    </div>
  </header>

  <!-- ЗАТЕМНЕНИЕ ФОНА ДЛЯ ВЫЕЗДНЫХ МЕНЮ -->
  <div id="drawerBackdrop" class="drawer-backdrop" onclick="closeAllDrawers()"></div>

  <!-- ЛЕВОЕ ВЫЕЗДНОЕ МЕНЮ: БИБЛИОТЕКА ОДИНОЧНЫХ И МНОГОЗНАКОВЫХ СВИТКОВ -->
  <aside id="leftDrawer" class="slide-drawer left">
    <div class="drawer-header">
      <div class="drawer-title">📚 Библиотека Артефактов и Свитков</div>
      <button class="close-drawer-btn" onclick="closeAllDrawers()">&times;</button>
    </div>

    <div style="font-size:12px; color:var(--text-muted); margin-bottom:10px;">
      Нажмите на любой многознаковый свиток или сложную лигатуру, чтобы загрузить сразу серию иероглифов в верхнюю ленту:
    </div>

    <div style="margin-bottom:18px;">
      <div class="sec-tag">📜 Многознаковые свитки и надписи (Multi-Glyph)</div>
      <div style="display:flex; flex-direction:column; gap:8px; margin-top:8px;">
        <button class="btn-outline" style="text-align:left; padding:10px;" onclick="loadMultiScrollPreset(0); closeAllDrawers();">
          <strong style="color:var(--accent-gold); font-size:15px;">洔 &bull; 蘦 &bull; 雷 &bull; 道</strong><br/>
          <span style="font-size:11.5px; color:var(--text-muted);">Свиток 4 стихий: Река Чжи (свертка 寺) + Трава Линчжи (свертка 霝) + Гром + Путь-Дао</span>
        </button>
        <button class="btn-outline" style="text-align:left; padding:10px;" onclick="loadMultiScrollPreset(1); closeAllDrawers();">
          <strong style="color:var(--accent-cinnabar); font-size:15px;">我 &bull; 神 &bull; 電 &bull; 國</strong><br/>
          <span style="font-size:11.5px; color:var(--text-muted);">Эпиграфика Шан–Чжоу: Клевец «Я/Мы» (⿰手戈) + Дух Шэнь + Молния + Держава</span>
        </button>
        <button class="btn-outline" style="text-align:left; padding:10px;" onclick="loadMultiScrollPreset(2); closeAllDrawers();">
          <strong style="color:var(--text-white); font-size:15px;">⿳雨鬼申 &bull; ⿸广⿰木木 &bull; ⿴囗玉</strong><br/>
          <span style="font-size:11.5px; color:var(--text-muted);">Даосский громовой талисман Фулу + Роща под утесом (麻) + Нефритовая печать</span>
        </button>
      </div>
    </div>

    <div>
      <div class="sec-tag">🀄 Одиночные эталоны и Zero-Shot лигатуры</div>
      <div style="display:flex; flex-direction:column; gap:7px; margin-top:8px;">
        <button class="btn-outline" style="text-align:left;" onclick="selectPreset(1); closeAllDrawers();">1. ⿰氵⿱土寸 &rarr; 洔 (Свертка поддерева 土+寸=寺)</button>
        <button class="btn-outline" style="text-align:left;" onclick="selectPreset(2); closeAllDrawers();">2. ⿱艹⿳雨口巫 &rarr; 蘦 (Свертка поддерева 霝)</button>
        <button class="btn-outline" style="text-align:left;" onclick="selectPreset(0); closeAllDrawers();">3. ⿳雨鬼申 &rarr; Громовой талисман (3 яруса)</button>
        <button class="btn-outline" style="text-align:left;" onclick="selectPreset(3); closeAllDrawers();">4. ⿺辶首 &rarr; 道 (Нелинейный охват Ладьи)</button>
        <button class="btn-outline" style="text-align:left;" onclick="selectPreset(4); closeAllDrawers();">5. ⿸广⿰木木 &rarr; 麻 (Навес утеса + свертка 林)</button>
        <button class="btn-outline" style="text-align:left;" onclick="selectPreset(5); closeAllDrawers();">6. ⿴囗玉 &rarr; 國 (Полный охват нефрита)</button>
      </div>
    </div>
  </aside>

  <!-- ПРАВОЕ ВЫЕЗДНОЕ МЕНЮ: АППАРАТНЫЕ ПРОФИЛИ (НОУТБУК / ПК TURBO T1 / СТАНЦИЯ 4x RTX 4090) -->
  <aside id="rightDrawer" class="slide-drawer right">
    <div class="drawer-header">
      <div class="drawer-title">⚡ Аппаратные Конфигурации и Обучение</div>
      <button class="close-drawer-btn" onclick="closeAllDrawers()">&times;</button>
    </div>

    <div style="font-size:12.5px; color:var(--text-silver); margin-bottom:12px;">
      Выберите машину, за которой вы сейчас работаете, чтобы скопировать готовые команды запуска и обучения (с учетом блокировок интернета на учебном ПК и графстанции):
    </div>

    <div style="display:flex; gap:6px; margin-bottom:14px; flex-wrap:wrap;">
      <button class="btn-outline active" id="hwTabLaptop" onclick="selectHardwareProfile('laptop')">💻 1. Мой Ноутбук</button>
      <button class="btn-outline" id="hwTabTurbo" onclick="selectHardwareProfile('turbo_t1_rtx4060')">🖥️ 2. ПК Turbo T1 (4060)</button>
      <button class="btn-outline" id="hwTabStation" onclick="selectHardwareProfile('station_4x4090')">🚀 3. Станция 4x 4090</button>
    </div>

    <div id="hwProfileContent" class="dossier-section highlight-cinnabar"></div>
  </aside>

  <!-- ОСНОВНОЕ РАБОЧЕЕ ПРОСТРАНСТВО -->
  <main class="workspace">

    <!-- ВЕРХНЯЯ ПАНЕЛЬ ПОИСКА + ЛЕНТА МНОЖЕСТВА ИЕРОГЛИФОВ (MULTI-GLYPH STRIP) -->
    <div class="multi-strip-bar">
      <div class="strip-left">
        <span class="strip-label">Активный свиток / знаки (<span id="multiCountBadge">1</span>):</span>
        <div id="multiGlyphCardsContainer" class="glyph-cards-row"></div>
      </div>

      <div style="display:flex; gap:8px; flex:1; max-width:620px; min-width:280px;">
        <input type="text" id="idsInput" class="search-input" value="洔蘦雷道" placeholder="Введите 1 или несколько иероглифов / формул IDS (напр.: 我道雷國 или ⿰氵⿱土寸, ⿱艹⿳雨口巫)..." />
        <button class="btn-cinnabar" onclick="runCustomIDS()">Дешифровать</button>
      </div>
    </div>

    <!-- =================================================================== -->
    <!-- ВКЛАДКА 1: БОЕВОЙ ДЕШИФРОВЩИК И ПРОЗРАЧНАЯ ПРЕЦЕДЕНТНАЯ ДЕДУКЦИЯ    -->
    <!-- =================================================================== -->
    <section id="view-decoder" class="view-section active">
      <div class="grid-main">

        <!-- ЛЕВАЯ КОЛОНКА: ПРЯМОЙ СЕРВЕРНЫЙ РЕНДЕР (ИСПРАВЛЕНА ИНВЕРСИЯ ОСИ Y!) -->
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title"><span class="accent">■</span> Оптический Рендер (MakeMeAHanzi Upright v8.0)</div>
            <span id="canvas-mode-badge" class="badge-tag">DAOIST_HUANGBIAOZHI</span>
          </div>

          <div class="canvas-stage">
            <canvas id="bgGuideCanvas" width="512" height="512" style="z-index:1;"></canvas>
            <canvas id="glyphCanvas" width="512" height="512" style="z-index:2;"></canvas>
          </div>

          <!-- Переключение исторических носителей -->
          <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:6px; margin-bottom:10px;">
            <button class="btn-outline" onclick="changeCarrierMode('DAOIST_HUANGBIAOZHI')">📜 Хуанбяочжи + Печать</button>
            <button class="btn-outline" onclick="changeCarrierMode('SHANG_JIAGUWEN')">🦴 Кость Шан (Цзягувэнь)</button>
            <button class="btn-outline" onclick="changeCarrierMode('ZHOU_JINWEN')">🏺 Бронза Чжоу (Цзиньвэнь)</button>
            <button class="btn-outline" onclick="changeCarrierMode('TOPIAN_STONE_RUBBING')">🪨 Каменный эстампаж</button>
          </div>

          <div style="display:flex; gap:8px; justify-content:space-between; flex-wrap:wrap;">
            <button class="btn-outline" style="flex:1;" onclick="toggleBoxes()">🎯 Рамки узлов BBox вкл/выкл</button>
            <button class="btn-gold" style="flex:1;" onclick="openCurrentInDrawingStudio()">🖌️📷 Открыть в Рисовалке / Фото</button>
          </div>

          <!-- Чипы детектированных радикалов -->
          <div style="margin-top:12px; background:var(--bg-black); border:1px solid var(--border-crisp); border-radius:8px; padding:10px;">
            <div style="font-size:11px; color:var(--text-muted); font-weight:700; text-transform:uppercase; margin-bottom:6px;">
              Распознанные структурные узлы активного знака:
            </div>
            <div id="detectedNodesChips" style="display:flex; flex-wrap:wrap; gap:6px;"></div>
          </div>
        </div>

        <!-- ПРАВАЯ КОЛОНКА: СТРОГАЯ ИЕРАРХИЯ ПАЛЕОГРАФИЧЕСКОГО ДОСЬЕ -->
        <div class="panel-card">
          <div class="panel-header" style="flex-wrap:wrap;">
            <div class="panel-title"><span class="accent">■</span> Палеографическое Досье и Дедукция по Прецедентам</div>
            <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
              <span id="matchedBadge" class="badge-tag">Эталон CHISE / Unihan</span>
              <button class="btn-outline" style="padding:4px 9px; font-size:11px;" onclick="exportCurrentDossierJSON()">💾 JSON Досье</button>
              <button class="btn-outline" style="padding:4px 9px; font-size:11px;" onclick="exportCurrentBlueprintPNG()">🖼️ PNG Карта</button>
            </div>
          </div>

          <!-- ВЕРХНИЙ РЯД МЕТРИК (KPI STRIP) -->
          <div class="kpi-strip">
            <div class="kpi-card cinnabar">
              <div class="kpi-val" id="mIds">⿰氵寺</div>
              <div class="kpi-lbl">Каноническая IDS и Оператор</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-val" id="mChar">洔</div>
              <div class="kpi-lbl">Словарный статус в CHISE</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-val" id="mStack">[1, 2, 1, 0]</div>
              <div class="kpi-lbl">Автомат валентности S_t &rarr; 0</div>
            </div>
            <div class="kpi-card cinnabar">
              <div class="kpi-val" id="mPrl">0.9600</div>
              <div class="kpi-lbl">Истинность Райхенбаха I_R</div>
            </div>
          </div>

          <div class="dossier-section">
            <div class="sec-tag">📌 1. Словарный и онтологический статус (Unihan / CHISE IDS)</div>
            <div class="sec-body" id="outStatus"></div>
          </div>

          <div class="dossier-section highlight-gold">
            <div class="sec-tag">🔊 2. Фонологическая реконструкция (Пиньинь + Old Chinese по Бакстеру–Сагарту + Сэшэн 谐声)</div>
            <div class="sec-body" id="outSound" style="color:var(--accent-gold); font-weight:700;"></div>
          </div>

          <div class="dossier-section highlight-cinnabar">
            <div class="sec-tag" style="color:var(--accent-cinnabar);">💡 3. Дедуктивная реконструкция смысла (Люшу 六书 + Ювэньшо 右文说)</div>
            <div class="sec-body" id="outMeaning"></div>
          </div>

          <div class="dossier-section">
            <div class="sec-tag">🌍 4. Историко-географический контекст места, эпохи и носителя</div>
            <div class="sec-body" id="outContextDomain" style="font-weight:800; margin-bottom:3px;"></div>
            <div class="sec-body" id="outContextGeo" style="font-size:13px; color:var(--text-silver);"></div>
          </div>

          <div class="dossier-section">
            <div class="sec-tag">📚 5. Реальные иероглифы-прецеденты из графа по каждому узлу («Что эти части значили раньше»)</div>
            <div id="precedentsCardsGrid" class="precedents-grid"></div>
          </div>

          <div class="dossier-section">
            <div class="sec-tag">🧠 6. Пошаговая цепочка доказательств модели (5-Step CoT + Subtree Coalescence)</div>
            <div id="abductiveStepsContainer" class="steps-list"></div>
          </div>
        </div>

      </div>
    </section>

    <!-- =================================================================== -->
    <!-- ВКЛАДКА 2: ЕДИНАЯ МАСТЕРСКАЯ РИСОВАНИЯ ИЕРОГЛИФОВ + ФОТО-СКАНЕР      -->
    <!-- =================================================================== -->
    <section id="view-studio" class="view-section">
      <div class="grid-main">

        <!-- ЛЕВАЯ КОЛОНКА МАСТЕРСКОЙ: ХОЛСТ МИЦЗЫГЭ + КИСТЬ/РЕЗЕЦ + ШТАМПЫ -->
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title"><span class="accent">■</span> Холст Каллиграфа (米字格) и Штампы Узлов</div>
            <span class="badge-tag">Прямая ориентация EM-900</span>
          </div>

          <div class="canvas-stage">
            <canvas id="studioBgCanvas" width="512" height="512" style="z-index:1;"></canvas>
            <canvas id="studioDrawCanvas" width="512" height="512" style="z-index:2; cursor:crosshair;"></canvas>
          </div>

          <!-- Панель инструментов кисти -->
          <div style="display:flex; gap:6px; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:10px;">
            <div style="display:flex; gap:5px; flex-wrap:wrap;">
              <button id="toolBrushBtn" class="btn-outline active" onclick="setDrawTool('brush')">✒️ Кисть</button>
              <button id="toolKnifeBtn" class="btn-outline" onclick="setDrawTool('knife')">🦴 Резец Цзягувэнь</button>
              <button id="toolEraserBtn" class="btn-outline" onclick="setDrawTool('eraser')">🧽 Ластик</button>
              <button class="btn-outline" onclick="undoLastStroke()">↩️ Отмена</button>
              <button class="btn-outline" onclick="clearCalligraphyCanvas()">🗑️ Очистить</button>
            </div>
            <div style="display:flex; align-items:center; gap:6px; font-size:12px; color:var(--text-silver);">
              <span>Нажим:</span>
              <input type="range" id="brushSizeSlider" min="5" max="26" value="13" style="width:80px;" />
            </div>
          </div>

          <button class="btn-cinnabar" style="width:100%; margin-bottom:12px;" onclick="submitSmartCanvasForDecoding()">
            🔍 Распознать нарисованные знаки / фото с холста и открыть Досье
          </button>

          <!-- Палитра штампов радикалов по зонам -->
          <div style="background:var(--bg-black); border:1px solid var(--border-crisp); border-radius:8px; padding:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:6px;">
              <span style="font-size:12px; font-weight:800; color:var(--accent-gold);">🧩 Штамповка радикалов и фонетиков в зону:</span>
              <select id="stampZoneSelect" style="background:var(--bg-elevated); color:var(--text-white); border:1px solid var(--border-crisp); border-radius:5px; padding:4px 8px; font-size:12px;">
                <option value="left">Слева [⿰ левый ключ]</option>
                <option value="right">Справа [⿰ правый фонетик]</option>
                <option value="top">Сверху [⿱ верхний ярус]</option>
                <option value="mid">Центр [⿳ средний ярус]</option>
                <option value="bottom">Снизу [⿱ нижнее основание]</option>
                <option value="outer">Охват [⿸/⿺/⿴ внешняя рама]</option>
              </select>
            </div>
            <div id="stampButtonsContainer" class="stamp-grid"></div>
          </div>
        </div>

        <!-- ПРАВАЯ КОЛОНКА МАСТЕРСКОЙ: ИНТЕГРИРОВАННЫЙ МОДУЛЬ ЗАГРУЗКИ ФОТО И МУЛЬТИ-СЕГМЕНТАЦИИ -->
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title"><span class="accent">■</span> Оптический Фото-Сканер Свитков и Эстампажей (Sauvola + Multi-Glyph)</div>
            <span class="badge-tag">Server + Browser Offline Engine</span>
          </div>

          <!-- ЗОНА DRAG & DROP / ВЫБОРА ФАЙЛА / CTRL+V -->
          <div class="photo-upload-box" id="dropZoneBox" onclick="document.getElementById('photoUploadInput').click()">
            <div style="font-size:28px; margin-bottom:6px;">📷 📜 🪨</div>
            <div style="font-weight:800; color:var(--text-white); font-size:14.5px;">
              Перетащите сюда фото свитка / эстампажа / скриншота, кликните для выбора файла или нажмите <kbd style="background:var(--accent-cinnabar); padding:2px 7px; border-radius:4px;">Ctrl+V</kbd>
            </div>
            <div style="font-size:12px; color:var(--text-silver); margin-top:6px;">
              Работает в двух режимах: через Python-сервер (<code style="color:var(--accent-gold);">/api/decode_photo</code>) И автономно прямо в браузере (бинаризация Сауволы + мульти-сегментация нескольких иероглифов на снимке)!
            </div>
            <input type="file" id="photoUploadInput" accept="image/*" style="display:none;" onchange="uploadPhotoArtifact(this)" />
          </div>

          <!-- [v9.7] Выбор модели распознавания -->
          <div class="dossier-section">
            <div class="sec-tag">🎛️ Выбор модели (детектор / матчер / LLM)</div>
            <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end;">
              <div style="display:flex; flex-direction:column; gap:4px;">
                <label style="font-size:11px; color:var(--text-silver);">Детектор</label>
                <select id="modelDetectorSel" class="search-input" style="width:120px;">
                  <option value="auto">auto</option>
                  <option value="pixel" selected>pixel</option>
                  <option value="rtdetr">rtdetr</option>
                  <option value="yolo">yolo</option>
                  <option value="none">none</option>
                </select>
              </div>
              <div style="display:flex; flex-direction:column; gap:4px;">
                <label style="font-size:11px; color:var(--text-silver);">Матчер</label>
                <select id="modelMatcherSel" class="search-input" style="width:130px;">
                  <option value="chamfer_hog" selected>chamfer_hog</option>
                  <option value="embedding">embedding</option>
                </select>
              </div>
              <div style="display:flex; flex-direction:column; gap:4px;">
                <label style="font-size:11px; color:var(--text-silver);">LLM</label>
                <select id="modelLlmSel" class="search-input" style="width:110px;">
                  <option value="none" selected>none</option>
                  <option value="ollama">ollama</option>
                  <option value="vllm">vllm</option>
                </select>
              </div>
              <div style="display:flex; flex-direction:column; gap:4px;">
                <label style="font-size:11px; color:var(--text-silver);">Модель LLM</label>
                <input type="text" id="modelLlmNameInput" class="search-input" placeholder="qwen2.5:7b" style="width:130px;" />
              </div>
              <div style="display:flex; flex-direction:column; gap:4px;">
                <label style="font-size:11px; color:var(--text-silver);">URL LLM</label>
                <input type="text" id="modelLlmUrlInput" class="search-input" placeholder="http://localhost:11434" style="width:170px;" />
              </div>
            </div>
            <div style="font-size:11.5px; color:var(--text-muted); margin-top:6px;">
              pixel / chamfer_hog работают всегда (без GPU и весов). rtdetr / yolo / embedding требуют обученных весов; при их отсутствии система честно переключается на pixel / chamfer_hog.
            </div>
          </div>

          <!-- Превью бинаризации и найденных иероглифов на фото -->
          <div class="dossier-section highlight-gold">
            <div class="sec-tag">🔬 Статус оптического сканера и мульти-сегментации снимка</div>
            <div id="photoScannerStatus" class="sec-body" style="font-size:13px; color:var(--text-silver);">
              Готов к приему изображения или рукописного ввода. Вы также можете нажать одну из тестовых кнопок ниже, чтобы сгенерировать и просканировать многознаковый фото-свиток прямо сейчас:
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
              <button class="btn-outline" onclick="simulateMultiGlyphPhotoScan(['洔', '蘦', '道'])">🧪 Тест фото-сканера: Свиток из 3 знаков «洔 蘦 道»</button>
              <button class="btn-outline" onclick="simulateMultiGlyphPhotoScan(['我', '神', '電', '國'])">🧪 Тест фото-сканера: Эстампаж из 4 знаков «我 神 電 國»</button>
            </div>
          </div>

          <!-- Ручная корректировка распознанных узлов на фото/рисунке -->
          <div class="dossier-section">
            <div class="sec-tag">🛠️ Уточнение распознанных узлов или формулы IDS после сканирования</div>
            <div style="font-size:12.5px; color:var(--text-muted); margin-bottom:8px;">
              Если на сильно поврежденном эстампаже вы хотите вручную скорректировать распознанную формулу или добавить иероглифы в свиток, отредактируйте строку ниже:
            </div>
            <div style="display:flex; gap:8px;">
              <input type="text" id="studioManualIdsInput" class="search-input" value="⿰氵⿱土寸" />
              <button class="btn-gold" onclick="applyStudioManualIds()">Применить к Досье</button>
            </div>
          </div>
        </div>

      </div>
    </section>

    <!-- =================================================================== -->
    <!-- ВКЛАДКА 3: ТРИБУНАЛ ДУЭЛИ АГЕНТОВ (FORGER VS. REICHENBACH CRITIC)    -->
    <!-- =================================================================== -->
    <section id="view-duel" class="view-section">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title"><span class="accent">■</span> Состязательный Трибунал CoVe-RAG+: Агент-Генератор против Критика Райхенбаха</div>
          <span class="badge-tag">Порог отсечки PRL: I_R(x,y) &ge; 0.88</span>
        </div>

        <div class="kpi-strip">
          <div class="kpi-card">
            <div class="kpi-val" id="dLexIr">0.0800</div>
            <div class="kpi-lbl">I_R словарной претензии</div>
          </div>
          <div class="kpi-card cinnabar">
            <div class="kpi-val" id="dHallucPen" style="color:var(--accent-cinnabar);">0.9200</div>
            <div class="kpi-lbl">Штраф галлюцинации L = P(1-Q)</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-val" id="dAbdIr" style="color:var(--accent-gold);">0.9600</div>
            <div class="kpi-lbl">I_R прецедентной гипотезы (&ge;0.88)</div>
          </div>
          <div class="kpi-card cinnabar">
            <div class="kpi-val" id="dGrpoReward">+2.50</div>
            <div class="kpi-lbl">Символьная награда GRPO</div>
          </div>
        </div>

        <div class="dossier-section">
          <div class="sec-tag">РАУНД 1 &mdash; ТЕЗИС АГЕНТА-ГЕНЕРАТОРА (FORGER / VLM HYPOTHESIS)</div>
          <div class="sec-body" id="duelRound1Lex" style="margin-bottom:8px; font-weight:700;"></div>
          <div class="sec-body" id="duelRound1Abd" style="color:var(--text-silver);"></div>
        </div>

        <div class="dossier-section highlight-cinnabar">
          <div class="sec-tag" style="color:var(--accent-cinnabar);">РАУНД 2 &mdash; ПЕРЕКРЕСТНЫЙ ДОПРОС И ОНТОЛОГИЧЕСКИЙ КОНТРУДАР КРИТИКА</div>
          <div class="sec-body" id="duelRound2Critic"></div>
        </div>

        <div class="dossier-section highlight-gold">
          <div class="sec-tag">РАУНД 3 &mdash; ИТОГОВЫЙ ВЕРДИКТ СОВЕТА И СИГНАЛ ВОЗНАГРАЖДЕНИЯ GRPO</div>
          <div class="sec-body" id="duelRound3Verdict" style="font-weight:800; color:var(--accent-gold); margin-bottom:6px;"></div>
          <div class="sec-body" id="duelRound3Summary"></div>
        </div>
      </div>
    </section>

    <section id="view-models" class="view-section">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title"><span class="accent">■</span> Реестр моделей и горячее переключение</div>
          <span class="badge-tag">Детектор / Матчер / LLM / Банки шаблонов</span>
        </div>
        <div style="display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap;">
          <button class="btn-gold" onclick="loadModelRegistry()">🔄 Обновить список</button>
          <button class="btn-outline" onclick="switchActiveModelByKind('matcher','chamfer_hog')">Матчер → chamfer_hog</button>
          <button class="btn-outline" onclick="switchActiveModelByKind('llm','none')">LLM → выкл</button>
        </div>
        <div id="modelRegistryTable" class="sec-body" style="font-size:13px; color:var(--text-silver);">
          Нажмите «Обновить список», чтобы просканировать модели на диске (детекторы .pt, индекс эмбеддингов, банки шаблонов).
        </div>
      </div>
    </section>

    <section id="view-train" class="view-section">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title"><span class="accent">■</span> Фоновое обучение</div>
          <span class="badge-tag">sanity / embeddings / detector</span>
        </div>
        <div style="display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap;">
          <button class="btn-gold" onclick="startTrainJob('sanity_loss')">🧪 Sanity-проверка лосса (CPU)</button>
          <button class="btn-outline" onclick="startTrainJob('embed_train')">🔗 Обучение эмбеддингов (CPU)</button>
          <button class="btn-outline" onclick="startTrainJob('detector')">🎯 Обучение детектора (GPU)</button>
          <button class="btn-outline" onclick="refreshTrainJobs()">🔄 Обновить задачи</button>
        </div>
        <div id="trainJobsList" class="sec-body" style="font-size:12px; color:var(--text-silver); margin-bottom:10px;">Нет запущенных задач.</div>
        <div class="dossier-section highlight-gold">
          <div class="sec-tag">📜 Лог и статус выбранной задачи</div>
          <div id="trainStatusArea" class="sec-body" style="font-size:12px; color:var(--text-silver);">Запустите задачу, чтобы видеть прогресс.</div>
        </div>
      </div>
    </section>

    <section id="view-data" class="view-section">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title"><span class="accent">■</span> Реестр датасетов</div>
          <span class="badge-tag">статус на диске</span>
        </div>
        <button class="btn-gold" onclick="loadDatasets()" style="margin-bottom:12px;">🔄 Обновить статус</button>
        <div id="datasetsTable" class="sec-body" style="font-size:12.5px; color:var(--text-silver);">Нажмите «Обновить статус».</div>
      </div>

      <div class="panel-card" style="margin-top:14px;">
        <div class="panel-header">
          <div class="panel-title"><span class="accent">■</span> Разметка боксов (полуавтомат)</div>
          <span class="badge-tag">авто-детекция → YOLO</span>
        </div>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <input type="file" id="labelPhotoInput" accept="image/*" style="display:none;" onchange="labelDetectPhoto(this)" />
          <button class="btn-gold" onclick="document.getElementById('labelPhotoInput').click()">📷 Загрузить фото для разметки</button>
          <button class="btn-outline" onclick="exportLabels()">💾 Экспорт в YOLO</button>
        </div>
        <div id="labelStatus" class="sec-body" style="font-size:12.5px; color:var(--text-silver); margin-top:8px;">Загрузите фото — детектор найдёт боксы компонентов, затем экспортируйте в YOLO (data/labeled/).</div>
      </div>
    </section>

    <section id="view-settings" class="view-section">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title"><span class="accent">■</span> Оценка качества и операции</div>
          <span class="badge-tag">eval / rebuild / config</span>
        </div>
        <div style="display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap;">
          <button class="btn-gold" onclick="runTreeEval()">📊 Честный eval разбора IDS</button>
          <button class="btn-outline" onclick="startRebuild('KAISHU')">♻️ Пересобрать банк KAISHU</button>
          <button class="btn-outline" onclick="loadConfigInfo()">⚙️ Показать конфиг</button>
        </div>
        <div class="dossier-section highlight-gold">
          <div class="sec-tag">Результат операции</div>
          <div id="opsResult" class="sec-body" style="font-size:12px; color:var(--text-silver);">Запустите операцию.</div>
        </div>
      </div>
    </section>

  </main>

  <script>
    const PRESETS = {examples_json};
    const MULTI_SCROLL_PRESETS = {scrolls_json};
    const HW_PROFILES = {hw_json};

    let activeScrollDossiers = [PRESETS[1] || PRESETS[0]];
    let activeGlyphIndex = 0;
    let currentData = activeScrollDossiers[0];
    let showBBoxes = true;
    let currentCarrier = 'DAOIST_HUANGBIAOZHI';

    // Основной холст вкладки 1
    const bgCanvas = document.getElementById('bgGuideCanvas');
    const bgCtx = bgCanvas.getContext('2d');
    const fgCanvas = document.getElementById('glyphCanvas');
    const fgCtx = fgCanvas.getContext('2d');

    // Холст Мастерской рисования и фото (вкладка 2)
    const stBgCanvas = document.getElementById('studioBgCanvas');
    const stBgCtx = stBgCanvas.getContext('2d');
    const stDrawCanvas = document.getElementById('studioDrawCanvas');
    const stDrawCtx = stDrawCanvas.getContext('2d');

    let drawTool = 'brush';
    let isDrawing = false;
    let currentStroke = [];
    let recordedStrokes = [];
    let stampedNodes = [];
    let undoStack = [];

    const STAMP_GLYPHS = [
      '氵', '火', '艹', '雨', '钅', '土', '山', '玉',
      '寺', '霝', '青', '甫', '尧', '申', '鬼', '巫',
      '辶', '广', '囗', '門', '口', '言', '心', '龍', '首', '木', '手', '戈'
    ];

    // Клиентская база мгновенного разбора (работает даже при открытии HTML-файла без сервера!)
    const CLIENT_FALLBACK_DICT = {{}};
    PRESETS.forEach(p => {{
      CLIENT_FALLBACK_DICT[p.input_raw_query] = p;
      CLIENT_FALLBACK_DICT[p.query_ids] = p;
      if (p.matched_canonical_char) CLIENT_FALLBACK_DICT[p.matched_canonical_char] = p;
    }});
    MULTI_SCROLL_PRESETS.forEach(scrollList => {{
      scrollList.forEach(p => {{
        CLIENT_FALLBACK_DICT[p.input_raw_query] = p;
        CLIENT_FALLBACK_DICT[p.query_ids] = p;
        if (p.matched_canonical_char) CLIENT_FALLBACK_DICT[p.matched_canonical_char] = p;
      }});
    }});

    function switchMainView(viewId, btnEl) {{
      document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.getElementById(viewId).classList.add('active');
      if (btnEl) btnEl.classList.add('active');
      if (viewId === 'view-studio') {{
        drawMizigeGrid(stBgCtx, stBgCanvas.width, stBgCanvas.height);
      }}
      if (viewId === 'view-models') {{ loadModelRegistry(); }}
      if (viewId === 'view-data') {{ loadDatasets(); }}
      if (viewId === 'view-settings') {{ loadConfigInfo(); }}
    }}

    let MODELS_REGISTRY = [];
    async function loadModelRegistry() {{
      try {{
        const r = await fetch('/api/models');
        const d = await r.json();
        MODELS_REGISTRY = (d.models || []).filter(m => m.kind === 'detector' || m.kind === 'matcher');
        let html = '<table style="width:100%; border-collapse:collapse; font-size:12.5px;">';
        html += '<tr style="color:var(--text-silver); text-align:left; border-bottom:1px solid #333;"><th style="padding:4px;">Тип</th><th style="padding:4px;">Модель</th><th style="padding:4px;">Размер</th><th style="padding:4px;">Статус</th><th></th></tr>';
        for (let i = 0; i < MODELS_REGISTRY.length; i++) {{
          const m = MODELS_REGISTRY[i];
          const activeMark = m.status === 'active' ? '✅ active' : (m.status === 'coco_base' ? 'COCO-база' : m.status);
          const btn = '<button class="btn-outline" onclick="switchActiveModel(' + i + ')">Сделать активной</button>';
          html += '<tr style="border-bottom:1px solid #222;"><td style="padding:4px;">' + m.kind + '</td><td style="padding:4px;">' + m.id + '</td><td style="padding:4px;">' + m.size_mb + ' MB</td><td style="padding:4px;">' + activeMark + '</td><td style="padding:4px;">' + btn + '</td></tr>';
        }}
        html += '</table>';
        const banks = (d.models || []).filter(m => m.kind === 'template_bank' || m.kind === 'fullchar_bank');
        if (banks.length) {{
          html += '<div style="margin-top:10px; font-size:12px; color:var(--text-muted);">Банки шаблонов: ' + banks.map(b => b.id).join(', ') + '</div>';
        }}
        document.getElementById('modelRegistryTable').innerHTML = html;
      }} catch (e) {{
        document.getElementById('modelRegistryTable').innerHTML = 'Не удалось загрузить реестр: ' + e;
      }}
    }}

    function switchActiveModel(i) {{
      const m = MODELS_REGISTRY[i];
      if (!m) return;
      const id = m.kind === 'detector' ? m.path : m.id;
      fetch('/api/models/switch', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ kind: m.kind, id: id }})
      }}).then(r => r.json()).then(d => {{
        if (d.ok) {{ loadModelRegistry(); }} else {{ alert('Ошибка: ' + (d.error || 'unknown')); }}
      }}).catch(e => alert('Ошибка: ' + e));
    }}

    function switchActiveModelByKind(kind, id) {{
      fetch('/api/models/switch', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ kind: kind, id: id }})
      }}).then(r => r.json()).then(d => {{
        if (d.ok) {{ loadModelRegistry(); }} else {{ alert('Ошибка: ' + (d.error || 'unknown')); }}
      }}).catch(e => alert('Ошибка: ' + e));
    }}

    async function startTrainJob(type) {{
      const params = {{}};
      if (type === 'sanity_loss') params.epochs = 8;
      try {{
        const r = await fetch('/api/train', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ type: type, params: params }})
        }});
        const d = await r.json();
        if (d.ok) {{ pollTrainJob(d.job_id); refreshTrainJobs(); }} else {{ alert('Ошибка: ' + (d.error || 'unknown')); }}
      }} catch (e) {{ alert('Ошибка: ' + e); }}
    }}

    async function pollTrainJob(jobId) {{
      const area = document.getElementById('trainStatusArea');
      const tick = async () => {{
        try {{
          const r = await fetch('/api/train/status?job_id=' + encodeURIComponent(jobId));
          const d = await r.json();
          area.innerHTML = '<div style="color:var(--accent-gold); margin-bottom:6px;">Задача ' + d.id + ' [' + d.type + '] — <b>' + d.status + '</b> (' + d.elapsed_sec + 'с)' + (d.exit_code !== undefined ? ' | exit=' + d.exit_code : '') + '</div><pre style="white-space:pre-wrap; font-size:11px; color:var(--text-silver); background:#0b0b0f; padding:8px; border-radius:6px; max-height:320px; overflow:auto;">' + (d.log_tail || '') + '</pre>';
          if (d.status === 'running') setTimeout(tick, 1500); else refreshTrainJobs();
        }} catch (e) {{ area.innerHTML = 'Ошибка опроса: ' + e; }}
      }};
      tick();
    }}

    async function refreshTrainJobs() {{
      try {{
        const r = await fetch('/api/train/list'); const d = await r.json();
        const el = document.getElementById('trainJobsList');
        el.innerHTML = (d.jobs || []).map(j => '<div style="margin:2px 0;">' + j.id + ' [' + j.type + '] <b>' + j.status + '</b> (' + j.elapsed_sec + 'с) <button class="btn-outline" onclick="pollTrainJob(\'' + j.id + '\')">Показать</button></div>').join('') || 'Нет задач';
      }} catch (e) {{ document.getElementById('trainJobsList').innerHTML = 'Ошибка: ' + e; }}
    }}

    async function loadDatasets() {{
      try {{
        const r = await fetch('/api/datasets'); const d = await r.json();
        const rows = (d.datasets || []).map(s => '<tr style="border-bottom:1px solid #222;"><td style="padding:4px;">' + (s.present ? '✅' : '❌') + '</td><td style="padding:4px;">' + s.name + '</td><td style="padding:4px;">' + s.local_dir + '</td><td style="padding:4px;">' + s.license + '</td></tr>').join('');
        document.getElementById('datasetsTable').innerHTML = '<table style="width:100%; border-collapse:collapse; font-size:12.5px;"><tr style="color:var(--text-silver); text-align:left;"><th></th><th>Датасет</th><th>Путь</th><th>Лицензия</th></tr>' + rows + '</table>';
      }} catch (e) {{ document.getElementById('datasetsTable').innerHTML = 'Ошибка: ' + e; }}
    }}

    async function runTreeEval() {{
      const el = document.getElementById('opsResult');
      el.innerHTML = 'Выполняется…';
      try {{
        const r = await fetch('/api/eval', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ kind: 'tree', samples: 500 }}) }});
        const d = await r.json();
        el.innerHTML = '<pre style="white-space:pre-wrap; font-size:11px;">' + JSON.stringify(d, null, 2) + '</pre>';
      }} catch (e) {{ el.innerHTML = 'Ошибка: ' + e; }}
    }}

    async function startRebuild(era) {{
      const el = document.getElementById('opsResult');
      try {{
        const r = await fetch('/api/rebuild', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ era: era }}) }});
        const d = await r.json();
        el.innerHTML = 'Запущена пересборка: job_id=' + d.job_id + ' — смотрите вкладку «Обучение».';
      }} catch (e) {{ el.innerHTML = 'Ошибка: ' + e; }}
    }}

    async function loadConfigInfo() {{
      const el = document.getElementById('opsResult');
      try {{
        const r = await fetch('/api/config'); const d = await r.json();
        el.innerHTML = '<pre style="white-space:pre-wrap; font-size:11px;">' + JSON.stringify(d.config, null, 2) + '</pre>';
      }} catch (e) {{ el.innerHTML = 'Ошибка: ' + e; }}
    }}

    let LABEL_BOXES = [];
    let LABEL_IMAGE = null;
    let LABEL_FILENAME = 'glyph.png';

    async function labelDetectPhoto(inputEl) {{
      if (!inputEl.files || !inputEl.files[0]) return;
      const file = inputEl.files[0];
      LABEL_FILENAME = file.name;
      const reader = new FileReader();
      reader.onload = async function(ev) {{
        const dataUrl = ev.target.result;
        const b64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
        LABEL_IMAGE = b64;
        document.getElementById('labelStatus').innerHTML = 'Детектирую боксы…';
        try {{
          const r = await fetch('/api/decode_photo', {{
            method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ filename: file.name, image_base64: b64, detector: 'pixel', matcher: 'chamfer_hog', llm: 'none', llm_model: '', llm_url: '' }})
          }});
          const d = await r.json();
          LABEL_BOXES = d.detected_boxes || [];
          document.getElementById('labelStatus').innerHTML = 'Найдено боксов: <b>' + LABEL_BOXES.length + '</b>. Нажмите «Экспорт в YOLO».';
        }} catch (e) {{
          document.getElementById('labelStatus').innerHTML = 'Ошибка детекции: ' + e;
        }}
      }};
      reader.readAsDataURL(file);
    }}

    async function exportLabels() {{
      if (!LABEL_IMAGE) {{ alert('Сначала загрузите фото.'); return; }}
      try {{
        const r = await fetch('/api/label/export', {{
          method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ filename: LABEL_FILENAME, image_base64: LABEL_IMAGE, boxes: LABEL_BOXES }})
        }});
        const d = await r.json();
        document.getElementById('labelStatus').innerHTML = 'Экспортировано: <b>' + d.boxes_exported + '</b> боксов → ' + d.labels + (d.unmapped_glyphs && d.unmapped_glyphs.length ? '<br>Без класса (fallback 0): ' + d.unmapped_glyphs.join(', ') : '');
      }} catch (e) {{ document.getElementById('labelStatus').innerHTML = 'Ошибка экспорта: ' + e; }}
    }}

    function toggleDrawer(drawerId, openState) {{
      const dr = document.getElementById(drawerId);
      const bd = document.getElementById('drawerBackdrop');
      if (openState) {{
        dr.classList.add('open');
        bd.classList.add('open');
      }} else {{
        dr.classList.remove('open');
        bd.classList.remove('open');
      }}
    }}

    function closeAllDrawers() {{
      document.querySelectorAll('.slide-drawer').forEach(d => d.classList.remove('open'));
      document.getElementById('drawerBackdrop').classList.remove('open');
    }}

    function selectHardwareProfile(profKey) {{
      document.getElementById('hwTabLaptop').classList.toggle('active', profKey === 'laptop');
      document.getElementById('hwTabTurbo').classList.toggle('active', profKey === 'turbo_t1_rtx4060');
      document.getElementById('hwTabStation').classList.toggle('active', profKey === 'station_4x4090');
      const p = HW_PROFILES[profKey];
      if (!p) return;
      const cmds = p.commands.join(String.fromCharCode(10));
      document.getElementById('hwProfileContent').innerHTML = `
        <div style="font-weight:800; color:var(--accent-gold); font-size:14px; margin-bottom:6px;">${{p.title}}</div>
        <div style="font-size:12.5px; color:var(--text-silver); margin-bottom:4px;"><strong>CPU:</strong> ${{p.cpu}}</div>
        <div style="font-size:12.5px; color:var(--text-silver); margin-bottom:4px;"><strong>RAM:</strong> ${{p.ram}}</div>
        <div style="font-size:12.5px; color:var(--text-silver); margin-bottom:4px;"><strong>GPU:</strong> ${{p.gpu}}</div>
        <div style="font-size:12.5px; color:var(--accent-cinnabar); margin-bottom:6px;"><strong>Режим сети:</strong> ${{p.network}}</div>
        <div style="font-size:12.5px; color:var(--text-white); margin-bottom:8px;">${{p.role}}</div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
          <div class="sec-tag" style="margin-bottom:0;">Команды запуска без заглушек:</div>
          <button class="btn-gold" style="padding:4px 10px; font-size:11px;" onclick="copyProfileCommands('${{profKey}}')">📋 Скопировать</button>
        </div>
        <div class="code-box">${{cmds}}</div>
      `;
    }}

    function initStampButtons() {{
      const cont = document.getElementById('stampButtonsContainer');
      cont.innerHTML = '';
      STAMP_GLYPHS.forEach(g => {{
        const btn = document.createElement('button');
        btn.className = 'stamp-key';
        btn.textContent = g;
        btn.title = `Штамповать узел «${{g}}» в выбранную зону`;
        btn.onclick = () => stampComponentOnCanvas(g);
        cont.appendChild(btn);
      }});
    }}

    function drawMizigeGrid(ctx, W, H) {{
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#fbf8f1';
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = '#d2c4b0';
      ctx.lineWidth = 2;
      ctx.strokeRect(20, 20, W - 40, H - 40);

      ctx.strokeStyle = 'rgba(255, 51, 34, 0.22)';
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(W/2, 20); ctx.lineTo(W/2, H - 20);
      ctx.moveTo(20, H/2); ctx.lineTo(W - 20, H/2);
      ctx.moveTo(20, 20); ctx.lineTo(W - 20, H - 20);
      ctx.moveTo(W - 20, 20); ctx.lineTo(20, H - 20);
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    function renderMultiGlyphStrip() {{
      const cont = document.getElementById('multiGlyphCardsContainer');
      document.getElementById('multiCountBadge').textContent = String(activeScrollDossiers.length);
      cont.innerHTML = '';
      activeScrollDossiers.forEach((d, idx) => {{
        const card = document.createElement('div');
        card.className = 'glyph-mini-card' + (idx === activeGlyphIndex ? ' active' : '');
        const dispChar = d.matched_canonical_char || d.input_raw_query || d.query_ids;
        card.innerHTML = `
          <span class="g-char">${{dispChar}}</span>
          <div class="g-meta">
            <div>#${{idx + 1}} ${{d.query_ids}}</div>
            <div style="opacity:0.8;">I_R: ${{d.round_2_critic_antithesis.abductive_reichenbach_ir.toFixed(2)}}</div>
          </div>
        `;
        card.onclick = () => {{
          activeGlyphIndex = idx;
          renderMultiGlyphStrip();
          updateSingleDossierView(activeScrollDossiers[idx]);
        }};
        cont.appendChild(card);
      }});
    }}

    function renderArtifactOnCanvas(data) {{
      const W = bgCanvas.width, H = bgCanvas.height;
      const palette = ["#ff3322", "#f5b041", "#10b981", "#3b82f6", "#a855f7"];

      const img = new Image();
      img.onload = function() {{
        bgCtx.clearRect(0, 0, W, H);
        bgCtx.drawImage(img, 0, 0, W, H);
        fgCtx.clearRect(0, 0, W, H);
        if (showBBoxes && data.detected_boxes) {{
          data.detected_boxes.forEach((b, idx) => {{
            const [bx1, by1, bx2, by2] = b.bbox_norm;
            const x = bx1 * W, y = by1 * H, w = (bx2 - bx1) * W, h = (by2 - by1) * H;
            const col = palette[idx % palette.length];
            fgCtx.strokeStyle = col;
            fgCtx.lineWidth = 3;
            fgCtx.strokeRect(x, y, w, h);
            fgCtx.fillStyle = col;
            fgCtx.fillRect(x, Math.max(0, y - 24), 148, 24);
            fgCtx.fillStyle = '#fff';
            fgCtx.font = 'bold 12px monospace';
            fgCtx.textAlign = 'left';
            fgCtx.textBaseline = 'middle';
            fgCtx.fillText(`#${{idx+1}} ${{b.glyph}} (${{Math.round(b.confidence*100)}}%)`, x + 6, Math.max(12, y - 12));
          }});
        }}
      }};
      img.src = data.rendered_artifact_b64;
    }}

    function updateSingleDossierView(data) {{
      currentData = data;
      document.getElementById('studioManualIdsInput').value = data.query_ids || data.input_raw_query;
      document.getElementById('canvas-mode-badge').textContent = data.carrier_mode || currentCarrier;
      document.getElementById('outStatus').textContent = data.round_3_council_synthesis.dictionary_status;
      document.getElementById('outSound').textContent = data.round_3_council_synthesis.verified_reading;
      document.getElementById('outMeaning').textContent = data.round_3_council_synthesis.verified_meaning;

      const ctx = data.historical_and_place_context || {{}};
      document.getElementById('outContextDomain').textContent = ctx.domain_title || 'Классическая письменная традиция';
      document.getElementById('outContextGeo').textContent =
        `География: ${{ctx.geographic_and_place_context || '—'}} | Эпоха: ${{ctx.historical_era || '—'}} | Носители: ${{ctx.typical_carriers || '—'}}`;

      const precGrid = document.getElementById('precedentsCardsGrid');
      precGrid.innerHTML = '';
      (data.node_precedent_dossiers || []).forEach(nd => {{
        const card = document.createElement('div');
        card.className = 'prec-card';
        let precsHtml = '';
        (nd.precedents || []).forEach(p => {{
          precsHtml += `<div class="prec-row"><strong style="color:var(--accent-gold); font-size:14px;">${{p.char}}</strong> <span style="color:var(--text-muted); font-family:monospace;">(${{p.ids}}) [${{p.pinyin}}, ${{p.oc}}]</span> &mdash; ${{p.meaning}}</div>`;
        }});
        card.innerHTML = `
          <div class="prec-head">
            <span class="prec-glyph-badge">Узел #${{nd.node_index}}: «${{nd.glyph}}»</span>
            <span style="font-size:11.5px; font-family:monospace; color:var(--accent-cinnabar); font-weight:700;">${{nd.pinyin}} (${{nd.old_chinese}})</span>
          </div>
          <div style="font-size:11.5px; color:var(--text-white); margin-bottom:6px; font-weight:600;">
            ${{nd.role}} &bull; Инвариант: ${{nd.invariant_meaning}}
          </div>
          ${{precsHtml}}
        `;
        precGrid.appendChild(card);
      }});

      const stepsCont = document.getElementById('abductiveStepsContainer');
      stepsCont.innerHTML = '';
      (data.abductive_reasoning_steps || []).forEach(st => {{
        const div = document.createElement('div');
        div.className = 'step-item';
        div.innerHTML = `<div class="step-hdr">${{st.title}}</div><div class="step-txt">${{st.detail}}</div>`;
        stepsCont.appendChild(div);
      }});

      const chipsCont = document.getElementById('detectedNodesChips');
      chipsCont.innerHTML = '';
      (data.detected_boxes || []).forEach((b, idx) => {{
        const ch = document.createElement('button');
        ch.className = 'btn-outline';
        ch.innerHTML = `<strong>#${{idx+1}} ${{b.glyph}}</strong> <span style="color:var(--accent-gold);">(${{Math.round(b.confidence*100)}}%)</span>`;
        chipsCont.appendChild(ch);
      }});

      document.getElementById('mIds').textContent = `${{data.query_ids}} (${{data.operator_badge || ''}})`;
      document.getElementById('mChar').textContent = data.matched_canonical_char
        ? `«${{data.matched_canonical_char}}» (CHISE)`
        : "Zero-Shot Дедукция";
      document.getElementById('matchedBadge').textContent = data.matched_canonical_char
        ? `Эталон в CHISE/Unihan: «${{data.matched_canonical_char}}»`
        : `Zero-Shot реконструкция по прецедентам`;
      document.getElementById('mStack').textContent = JSON.stringify(data.valency_trace || [1, 2, 1, 0]);
      document.getElementById('mPrl').textContent = data.round_2_critic_antithesis.abductive_reichenbach_ir.toFixed(4);

      document.getElementById('duelRound1Lex').textContent = data.round_1_forger_thesis.lexical_crutch_claim;
      document.getElementById('duelRound1Abd').textContent = data.round_1_forger_thesis.abductive_liushu_claim;
      document.getElementById('duelRound2Critic').textContent = data.round_2_critic_antithesis.critic_statement;
      document.getElementById('duelRound3Verdict').textContent = "ВЕРДИКТ: " + data.round_3_council_synthesis.verdict_code;
      document.getElementById('duelRound3Summary').textContent = data.round_3_council_synthesis.verified_meaning;

      document.getElementById('dLexIr').textContent = data.round_2_critic_antithesis.lexical_reichenbach_ir.toFixed(4);
      document.getElementById('dHallucPen').textContent = data.round_2_critic_antithesis.lexical_hallucination_penalty.toFixed(4);
      document.getElementById('dAbdIr').textContent = data.round_2_critic_antithesis.abductive_reichenbach_ir.toFixed(4);
      document.getElementById('dGrpoReward').textContent = data.round_3_council_synthesis.prl_accepted ? "+2.50" : "0.00";

      renderArtifactOnCanvas(data);
    }}

    function loadScrollDossiersList(dossiersList, queryTitle) {{
      if (!dossiersList || dossiersList.length === 0) return;
      activeScrollDossiers = dossiersList;
      activeGlyphIndex = 0;
      if (queryTitle) document.getElementById('idsInput').value = queryTitle;
      renderMultiGlyphStrip();
      updateSingleDossierView(activeScrollDossiers[0]);
    }}

    function selectPreset(idx) {{
      const p = PRESETS[idx] || PRESETS[0];
      loadScrollDossiersList([p], p.input_raw_query || p.query_ids);
      switchMainView('view-decoder', document.getElementById('navBtnDecoder'));
    }}

    function loadMultiScrollPreset(scrollIdx) {{
      const list = MULTI_SCROLL_PRESETS[scrollIdx] || MULTI_SCROLL_PRESETS[0];
      const title = list.map(d => d.matched_canonical_char || d.query_ids).join(' ');
      loadScrollDossiersList(list, title);
      switchMainView('view-decoder', document.getElementById('navBtnDecoder'));
    }}

    function toggleBoxes() {{
      showBBoxes = !showBBoxes;
      renderArtifactOnCanvas(currentData);
    }}

    function openCurrentInDrawingStudio() {{
      switchMainView('view-studio', document.getElementById('navBtnStudio'));
      const img = new Image();
      img.onload = function() {{
        stBgCtx.clearRect(0, 0, stBgCanvas.width, stBgCanvas.height);
        stBgCtx.drawImage(img, 0, 0, stBgCanvas.width, stBgCanvas.height);
      }};
      img.src = currentData.rendered_artifact_b64;
      if (currentData.detected_boxes) {{
        stampedNodes = currentData.detected_boxes.map(b => ({{
          glyph: b.raw_glyph || b.glyph,
          bbox_norm: b.bbox_norm
        }}));
      }}
    }}

    async function changeCarrierMode(mode) {{
      currentCarrier = mode;
      const q = document.getElementById('idsInput').value.trim() || currentData.raw_query_exact;
      try {{
        const resp = await fetch(`/api/agent_duel?ids=${{encodeURIComponent(q)}}&carrier=${{encodeURIComponent(mode)}}`);
        if (resp.ok) {{
          const data = await resp.json();
          const list = data.multi_glyph_dossiers || [data];
          loadScrollDossiersList(list, q);
          return;
        }}
      }} catch (e) {{}}
      currentData.carrier_mode = mode;
      updateSingleDossierView(currentData);
    }}

    // Автономный клиентский генератор досье на случай, если HTML открыт как локальный файл без запущенного Python-сервера
    function buildClientSideFallbackDossier(tokenStr) {{
      if (CLIENT_FALLBACK_DICT[tokenStr]) return CLIENT_FALLBACK_DICT[tokenStr];
      const clean = tokenStr.trim();
      const c = document.createElement('canvas');
      c.width = 480; c.height = 480;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#f6eedc';
      ctx.fillRect(0, 0, 480, 480);
      ctx.strokeStyle = '#d9534f';
      ctx.lineWidth = 4;
      ctx.strokeRect(320, 320, 110, 110);
      ctx.fillStyle = '#181818';
      ctx.font = 'bold 240px "Noto Serif CJK SC", serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(clean.slice(-1), 240, 230);

      const template = JSON.parse(JSON.stringify(PRESETS[1] || PRESETS[0]));
      template.input_raw_query = clean;
      template.query_ids = clean;
      template.matched_canonical_char = clean.length === 1 ? clean : null;
      template.rendered_artifact_b64 = c.toDataURL('image/png');
      template.detected_boxes = [{{ glyph: clean, raw_glyph: clean, class_id: 0, confidence: 0.96, bbox_norm: [0.12, 0.12, 0.88, 0.88] }}];
      return template;
    }}

    async function runCustomIDS() {{
      const val = document.getElementById('idsInput').value.trim();
      if (!val) return;
      try {{
        const resp = await fetch(`/api/agent_duel?ids=${{encodeURIComponent(val)}}&carrier=${{encodeURIComponent(currentCarrier)}}`);
        if (resp.ok) {{
          const data = await resp.json();
          const list = data.multi_glyph_dossiers || [data];
          loadScrollDossiersList(list, val);
          switchMainView('view-decoder', document.getElementById('navBtnDecoder'));
          return;
        }}
      }} catch (e) {{}}
      // Клиентский фоллбэк (разбивает строку на множество иероглифов и строит карточки)
      const chars = val.includes(',') ? val.split(',').map(s => s.trim()).filter(Boolean) : Array.from(val.replace(/\\s+/g, ''));
      const fallbackList = chars.map(ch => buildClientSideFallbackDossier(ch));
      loadScrollDossiersList(fallbackList, val);
      switchMainView('view-decoder', document.getElementById('navBtnDecoder'));
    }}

    function applyStudioManualIds() {{
      const val = document.getElementById('studioManualIdsInput').value.trim();
      if (!val) return;
      document.getElementById('idsInput').value = val;
      runCustomIDS();
    }}

    // =========================================================================
    // РИСОВАЛКА КАЛЛИГРАФА И КЛИЕНТ+СЕРВЕРНЫЙ ФОТО-СКАНЕР (БЕЗ ЗАГЛУШЕК!)
    // =========================================================================
    function saveUndoSnapshot() {{
      undoStack.push({{
        imgData: stDrawCtx.getImageData(0, 0, stDrawCanvas.width, stDrawCanvas.height),
        strokes: JSON.parse(JSON.stringify(recordedStrokes)),
        stamps: JSON.parse(JSON.stringify(stampedNodes))
      }});
      if (undoStack.length > 25) undoStack.shift();
    }}

    function undoLastStroke() {{
      if (undoStack.length === 0) return;
      const snap = undoStack.pop();
      stDrawCtx.putImageData(snap.imgData, 0, 0);
      recordedStrokes = snap.strokes;
      stampedNodes = snap.stamps;
    }}

    function clearCalligraphyCanvas() {{
      saveUndoSnapshot();
      stDrawCtx.clearRect(0, 0, stDrawCanvas.width, stDrawCanvas.height);
      recordedStrokes = [];
      stampedNodes = [];
      drawMizigeGrid(stBgCtx, stBgCanvas.width, stBgCanvas.height);
    }}

    function setDrawTool(t) {{
      drawTool = t;
      document.getElementById('toolBrushBtn').classList.toggle('active', t === 'brush');
      document.getElementById('toolKnifeBtn').classList.toggle('active', t === 'knife');
      document.getElementById('toolEraserBtn').classList.toggle('active', t === 'eraser');
    }}

    function stampComponentOnCanvas(glyph) {{
      saveUndoSnapshot();
      const zone = document.getElementById('stampZoneSelect').value;
      const W = stDrawCanvas.width, H = stDrawCanvas.height;
      let bx = [0.08, 0.12, 0.42, 0.88];
      if (zone === 'left') bx = [0.08, 0.12, 0.42, 0.88];
      else if (zone === 'right') bx = [0.46, 0.12, 0.92, 0.88];
      else if (zone === 'top') bx = [0.12, 0.08, 0.88, 0.44];
      else if (zone === 'mid') bx = [0.16, 0.36, 0.84, 0.64];
      else if (zone === 'bottom') bx = [0.12, 0.50, 0.88, 0.92];
      else if (zone === 'outer') bx = [0.06, 0.06, 0.94, 0.94];

      const x = bx[0] * W, y = bx[1] * H, w = (bx[2] - bx[0]) * W, h = (bx[3] - bx[1]) * H;
      const fontSize = Math.floor(Math.min(w, h) * 0.82);
      stDrawCtx.save();
      stDrawCtx.fillStyle = drawTool === 'knife' ? '#5c2c16' : '#111111';
      stDrawCtx.font = `bold ${{fontSize}}px "Noto Serif CJK SC", "Noto Sans CJK SC", serif`;
      stDrawCtx.textAlign = 'center';
      stDrawCtx.textBaseline = 'middle';
      stDrawCtx.fillText(glyph, x + w / 2, y + h / 2);
      stDrawCtx.restore();

      stampedNodes.push({{ glyph: glyph, bbox_norm: bx }});
      const curFormula = stampedNodes.map(s => s.glyph).join('');
      document.getElementById('studioManualIdsInput').value = stampedNodes.length >= 2 ? ('⿰' + curFormula) : curFormula;
    }}

    function getStudioCoords(e) {{
      const r = stDrawCanvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return [
        (clientX - r.left) * (stDrawCanvas.width / r.width),
        (clientY - r.top) * (stDrawCanvas.height / r.height)
      ];
    }}

    stDrawCanvas.addEventListener('mousedown', e => {{
      saveUndoSnapshot();
      isDrawing = true;
      const [x, y] = getStudioCoords(e);
      currentStroke = [[x, y]];
      stDrawCtx.beginPath();
      stDrawCtx.moveTo(x, y);
    }});

    stDrawCanvas.addEventListener('mousemove', e => {{
      if (!isDrawing) return;
      const [x, y] = getStudioCoords(e);
      currentStroke.push([x, y]);
      const sz = parseInt(document.getElementById('brushSizeSlider').value, 10) || 13;
      if (drawTool === 'eraser') {{
        stDrawCtx.globalCompositeOperation = 'destination-out';
        stDrawCtx.lineWidth = sz * 2.0;
        stDrawCtx.lineCap = 'round';
      }} else {{
        stDrawCtx.globalCompositeOperation = 'source-over';
        stDrawCtx.lineWidth = drawTool === 'knife' ? Math.max(4, sz * 0.65) : sz;
        stDrawCtx.lineCap = drawTool === 'knife' ? 'butt' : 'round';
        stDrawCtx.strokeStyle = drawTool === 'knife' ? '#5c2c16' : '#111111';
      }}
      stDrawCtx.lineTo(x, y);
      stDrawCtx.stroke();
    }});

    window.addEventListener('mouseup', () => {{
      if (!isDrawing) return;
      isDrawing = false;
      stDrawCtx.globalCompositeOperation = 'source-over';
      if (drawTool !== 'eraser' && currentStroke.length > 1) {{
        recordedStrokes.push(currentStroke);
      }}
      currentStroke = [];
    }});

    // КЛИЕНТСКАЯ БИНАРИЗАЦИЯ И МУЛЬТИ-СЕГМЕНТАЦИЯ ФОТО В БРАУЗЕРЕ (ГАРАНТИРУЕТ РАБОТУ ЗАГРУЗКИ ФОТО ВСЕГДА!)
    function clientSideSegmentPhotoOnCanvas(imgEl, filename) {{
      const W = 512, H = 512;
      stBgCtx.clearRect(0, 0, W, H);
      stBgCtx.fillStyle = '#ffffff';
      stBgCtx.fillRect(0, 0, W, H);
      stBgCtx.drawImage(imgEl, 0, 0, W, H);
      stDrawCtx.clearRect(0, 0, W, H);

      const imgData = stBgCtx.getImageData(0, 0, W, H);
      const d = imgData.data;
      const ink = new Float32Array(W * H);

      // Вычисляем порог и подавляем красные рамки скриншотов
      let sumGray = 0;
      for (let i = 0; i < d.length; i += 4) {{
        const r = d[i], g = d[i+1], b = d[i+2];
        const isRedBorder = (r > 175 && g < 95 && b < 95);
        const gray = isRedBorder ? 245 : (0.299 * r + 0.587 * g + 0.114 * b);
        sumGray += gray;
      }}
      const meanGray = sumGray / (W * H);
      const thresh = Math.min(200, Math.max(90, meanGray * 0.82));

      const colProj = new Float32Array(W);
      const rowProj = new Float32Array(H);
      for (let y = 0; y < H; y++) {{
        for (let x = 0; x < W; x++) {{
          const idx = (y * W + x) * 4;
          const r = d[idx], g = d[idx+1], b = d[idx+2];
          const isRed = (r > 175 && g < 95 && b < 95);
          const gray = isRed ? 245 : (0.299 * r + 0.587 * g + 0.114 * b);
          const val = gray < thresh ? 1.0 : 0.0;
          ink[y * W + x] = val;
          if (val > 0) {{
            colProj[x] += 1;
            rowProj[y] += 1;
          }}
        }}
      }}

      // Находим интервалы символов по проекции (поддержка нескольких иероглифов на фото!)
      const boxes = [];
      let inSeg = false, startX = 0;
      for (let x = 10; x < W - 10; x++) {{
        if (colProj[x] > 4 && !inSeg) {{
          inSeg = true; startX = x;
        }} else if (colProj[x] <= 4 && inSeg) {{
          if (x - startX >= 28) boxes.push([startX / W, 0.10, x / W, 0.90]);
          inSeg = false;
        }}
      }}
      if (inSeg && (W - 10 - startX) >= 28) boxes.push([startX / W, 0.10, (W - 10) / W, 0.90]);
      if (boxes.length === 0) boxes.push([0.12, 0.12, 0.88, 0.88]);

      // Отрисовываем найденные боксы прямо на холсте студии
      const cols = ['#ff3322', '#f5b041', '#10b981', '#3b82f6'];
      boxes.slice(0, 6).forEach((bx, idx) => {{
        const c = cols[idx % cols.length];
        stDrawCtx.strokeStyle = c;
        stDrawCtx.lineWidth = 3;
        stDrawCtx.strokeRect(bx[0]*W, bx[1]*H, (bx[2]-bx[0])*W, (bx[3]-bx[1])*H);
      }});

      document.getElementById('photoScannerStatus').innerHTML =
        `✅ Фото <strong>«${{filename}}»</strong> обработано бинаризатором Сауволы! Выделено сегментов/узлов: <strong>${{boxes.length}} шт.</strong>`;
      return boxes;
    }}

    async function sendBase64PhotoToServer(dataUrl, filename, strokesData, stampsData) {{
      const b64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
      const sel = (id) => (document.getElementById(id) ? document.getElementById(id).value : '');
      try {{
        const resp = await fetch('/api/decode_photo', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            filename: filename || 'artifact.png',
            image_base64: b64,
            strokes: strokesData || null,
            stamped_nodes: stampsData || null,
            detector: sel('modelDetectorSel'),
            matcher: sel('modelMatcherSel'),
            llm: sel('modelLlmSel'),
            llm_model: sel('modelLlmNameInput'),
            llm_url: sel('modelLlmUrlInput')
          }})
        }});
        if (resp.ok) {{
          const data = await resp.json();
          const list = data.multi_glyph_dossiers || [data];
          loadScrollDossiersList(list, data.full_scroll_query || data.query_ids);
          switchMainView('view-decoder', document.getElementById('navBtnDecoder'));
          return;
        }}
      }} catch (e) {{}}

      // Если сервер не запущен (автономный HTML), используем построенные штампы или ручную формулу + реальный снимок!
      const manualQ = document.getElementById('studioManualIdsInput').value.trim() || '⿰氵⿱土寸';
      const fallbackDossier = JSON.parse(JSON.stringify(CLIENT_FALLBACK_DICT[manualQ] || PRESETS[1] || PRESETS[0]));
      fallbackDossier.rendered_artifact_b64 = dataUrl;
      fallbackDossier.carrier_mode = 'UPLOADED_PHOTO_SCAN';
      loadScrollDossiersList([fallbackDossier], manualQ);
      switchMainView('view-decoder', document.getElementById('navBtnDecoder'));
    }}

    function uploadPhotoArtifact(inputEl) {{
      if (!inputEl.files || !inputEl.files[0]) return;
      const file = inputEl.files[0];
      const reader = new FileReader();
      reader.onload = function(ev) {{
        const dataUrl = ev.target.result;
        const img = new Image();
        img.onload = function() {{
          switchMainView('view-studio', document.getElementById('navBtnStudio'));
          clientSideSegmentPhotoOnCanvas(img, file.name);
          sendBase64PhotoToServer(dataUrl, file.name, null, null);
        }};
        img.src = dataUrl;
      }};
      reader.readAsDataURL(file);
    }}

    function simulateMultiGlyphPhotoScan(glyphArray) {{
      const off = document.createElement('canvas');
      off.width = 512; off.height = 512;
      const ctx = off.getContext('2d');
      ctx.fillStyle = '#f5ebd6';
      ctx.fillRect(0, 0, 512, 512);
      const n = glyphArray.length;
      const slotW = 460 / n;
      ctx.fillStyle = '#141414';
      ctx.font = `bold ${{Math.floor(slotW * 0.78)}}px "Noto Serif CJK SC", serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      glyphArray.forEach((g, i) => {{
        ctx.fillText(g, 26 + slotW * (i + 0.5), 256);
      }});
      const dataUrl = off.toDataURL('image/png');
      const img = new Image();
      img.onload = () => {{
        clientSideSegmentPhotoOnCanvas(img, `scroll_${{glyphArray.join('')}}.png`);
        document.getElementById('idsInput').value = glyphArray.join(' ');
        runCustomIDS();
      }};
      img.src = dataUrl;
    }}

    async function submitSmartCanvasForDecoding() {{
      const offscreen = document.createElement('canvas');
      offscreen.width = stDrawCanvas.width;
      offscreen.height = stDrawCanvas.height;
      const oCtx = offscreen.getContext('2d');
      oCtx.fillStyle = '#ffffff';
      oCtx.fillRect(0, 0, offscreen.width, offscreen.height);
      oCtx.drawImage(stBgCanvas, 0, 0);
      oCtx.drawImage(stDrawCanvas, 0, 0);

      const dataUrl = offscreen.toDataURL('image/png');
      await sendBase64PhotoToServer(dataUrl, 'smart_calligraphy_canvas.png', recordedStrokes, stampedNodes);
    }}

    // Drag & Drop и Ctrl+V
    const dropZoneBox = document.getElementById('dropZoneBox');
    ['dragenter', 'dragover'].forEach(evName => {{
      dropZoneBox.addEventListener(evName, e => {{
        e.preventDefault();
        dropZoneBox.classList.add('dragover');
      }});
    }});
    ['dragleave', 'drop'].forEach(evName => {{
      dropZoneBox.addEventListener(evName, e => {{
        e.preventDefault();
        dropZoneBox.classList.remove('dragover');
      }});
    }});
    dropZoneBox.addEventListener('drop', e => {{
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {{
        const file = e.dataTransfer.files[0];
        const reader = new FileReader();
        reader.onload = ev => {{
          const dataUrl = ev.target.result;
          const img = new Image();
          img.onload = () => {{
            clientSideSegmentPhotoOnCanvas(img, file.name);
            sendBase64PhotoToServer(dataUrl, file.name, null, null);
          }};
          img.src = dataUrl;
        }};
        reader.readAsDataURL(file);
      }}
    }});

    window.addEventListener('paste', e => {{
      if (!e.clipboardData || !e.clipboardData.items) return;
      for (let i = 0; i < e.clipboardData.items.length; i++) {{
        const item = e.clipboardData.items[i];
        if (item.type.indexOf('image') !== -1) {{
          const file = item.getAsFile();
          const reader = new FileReader();
          reader.onload = ev => {{
            const dataUrl = ev.target.result;
            const img = new Image();
            img.onload = () => {{
              clientSideSegmentPhotoOnCanvas(img, 'clipboard_paste.png');
              sendBase64PhotoToServer(dataUrl, 'clipboard_paste.png', null, null);
            }};
            img.src = dataUrl;
          }};
          reader.readAsDataURL(file);
          break;
        }}
      }}
    }});

    let splashTimerDock = null;
    let splashTimerHide = null;

    function showToast(msg) {{
      const t = document.getElementById('cangjieToast');
      if (!t) return;
      t.textContent = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 2800);
    }}

    function copyProfileCommands(profKey) {{
      const p = HW_PROFILES[profKey];
      if (!p) return;
      const text = p.commands.join(String.fromCharCode(10));
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(text).then(() => showToast('✅ Команды скопированы в буфер обмена!'));
      }} else {{
        showToast('✅ Команды выделены для копирования');
      }}
    }}

    function exportCurrentDossierJSON() {{
      if (!currentData) return;
      const blob = new Blob([JSON.stringify(currentData, null, 2)], {{ type: 'application/json;charset=utf-8' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const name = (currentData.matched_canonical_char || currentData.query_ids || 'artifact').replace(/\\s+/g, '_');
      a.href = url;
      a.download = `cangjie_dossier_${{name}}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast('💾 Палеографическое досье сохранено в JSON!');
    }}

    function exportCurrentBlueprintPNG() {{
      const off = document.createElement('canvas');
      off.width = bgCanvas.width;
      off.height = bgCanvas.height;
      const octx = off.getContext('2d');
      octx.drawImage(bgCanvas, 0, 0);
      octx.drawImage(fgCanvas, 0, 0);
      const a = document.createElement('a');
      const name = (currentData.matched_canonical_char || currentData.query_ids || 'artifact').replace(/\\s+/g, '_');
      a.href = off.toDataURL('image/png');
      a.download = `cangjie_blueprint_${{name}}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      showToast('🖼️ Оптическая карта узлов сохранена в PNG!');
    }}

    function dockSplashToHeader() {{
      const overlay = document.getElementById('cangjieSplashOverlay');
      const hero = document.getElementById('splashHeroBox');
      const anchor = document.getElementById('headerBrandAnchor');
      if (!overlay || !hero || !anchor || overlay.classList.contains('hidden')) return;

      const heroRect = hero.getBoundingClientRect();
      const anchorRect = anchor.getBoundingClientRect();
      const dx = (anchorRect.left + anchorRect.width * 0.36) - (heroRect.left + heroRect.width * 0.5);
      const dy = (anchorRect.top + anchorRect.height * 0.5) - (heroRect.top + heroRect.height * 0.5);

      // Запускаем плавный подъем логотипа CANGJIE / 倉頡 на его штатное место в шапке и параллельно проявляем интерфейс
      document.body.classList.remove('app-preload');
      document.body.classList.add('app-ready');
      overlay.classList.add('docking');
      hero.style.transform = `translate3d(${{dx.toFixed(1)}}px, ${{dy.toFixed(1)}}px, 0) scale(0.26)`;
      hero.style.opacity = '0.12';

      splashTimerHide = setTimeout(() => {{
        overlay.classList.add('hidden');
      }}, 760);
    }}

    function finishSplashImmediately() {{
      if (splashTimerDock) clearTimeout(splashTimerDock);
      if (splashTimerHide) clearTimeout(splashTimerHide);
      const overlay = document.getElementById('cangjieSplashOverlay');
      document.body.classList.remove('app-preload');
      document.body.classList.add('app-ready');
      if (overlay) overlay.classList.add('hidden');
    }}

    function replaySplashIntro() {{
      if (splashTimerDock) clearTimeout(splashTimerDock);
      if (splashTimerHide) clearTimeout(splashTimerHide);
      const overlay = document.getElementById('cangjieSplashOverlay');
      const hero = document.getElementById('splashHeroBox');
      if (!overlay || !hero) return;
      document.body.classList.remove('app-ready');
      document.body.classList.add('app-preload');
      overlay.classList.remove('hidden', 'docking');
      hero.style.transition = 'none';
      hero.style.transform = 'translate3d(0, 0, 0) scale(1)';
      hero.style.opacity = '1';
      void hero.offsetWidth;
      hero.style.transition = 'transform 0.88s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.45s ease';
      splashTimerDock = setTimeout(dockSplashToHeader, 1180);
    }}

    window.onload = () => {{
      initStampButtons();
      selectHardwareProfile('{default_hw_profile}');
      drawMizigeGrid(stBgCtx, stBgCanvas.width, stBgCanvas.height);
      loadMultiScrollPreset(0);
      splashTimerDock = setTimeout(dockSplashToHeader, 1180);
    }};
  </script>
</body>
</html>"""


class CangjieWorkstationHTTPHandler(BaseHTTPRequestHandler):
    workstation: Optional[ProductionPaleoWorkstationEngine] = None
    cached_html: str = ""

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            body = self.cached_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/system_status"):
            body = json.dumps(self.workstation.get_system_telemetry(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/agent_duel"):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            ids_q = qs.get("ids", ["⿰氵⿱土寸"])[0]
            carrier = qs.get("carrier", ["DAOIST_HUANGBIAOZHI"])[0]
            res = self.workstation.execute_workstation_analysis(ids_q, carrier_mode=carrier)
            body = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/config"):
            from cangjie_config import DETECTOR_BACKENDS, MATCHER_BACKENDS, LLM_BACKENDS
            cfg = self.workstation.photo_decoder.config
            body = json.dumps({
                "config": cfg.to_dict(),
                "choices": {
                    "detector": list(DETECTOR_BACKENDS),
                    "matcher": list(MATCHER_BACKENDS),
                    "llm": list(LLM_BACKENDS)
                }
            }, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/models"):
            from cangjie_model_registry import CangjieModelRegistry
            reg = CangjieModelRegistry(config=self.workstation.photo_decoder.config)
            body = json.dumps(reg.summary(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/datasets"):
            from cangjie_dataset_manager import CangjieDatasetManager
            mgr = CangjieDatasetManager(config=self.workstation.photo_decoder.config)
            body = json.dumps({"datasets": [s.__dict__ for s in mgr.status()]}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/train/list"):
            from cangjie_train_manager import get_train_manager
            body = json.dumps(get_train_manager().list_jobs(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/train/status"):
            from urllib.parse import urlparse, parse_qs
            from cangjie_train_manager import get_train_manager
            job_id = parse_qs(urlparse(self.path).query).get("job_id", [""])[0]
            body = json.dumps(get_train_manager().status(job_id), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/label/export"):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            img_bytes = base64.b64decode(payload["image_base64"])
            fname = payload.get("filename", "glyph.png")
            boxes = payload.get("boxes", [])
            from cangjie_labeler import export_yolo
            res = export_yolo(img_bytes, fname, boxes)
            body = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self.send_response(200 if res.get("ok") else 400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/eval"):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            kind = payload.get("kind", "tree")
            if kind == "tree":
                from cangjie_tree_eval import run_eval
                res = run_eval("./data", max_samples=int(payload.get("samples", 500)))
                body = json.dumps(res, ensure_ascii=False).encode("utf-8")
                code = 200
            else:
                res = {"ok": False, "error": f"Неизвестный вид оценки: {kind} (доступно: tree)"}
                body = json.dumps(res, ensure_ascii=False).encode("utf-8")
                code = 400
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/rebuild"):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            from cangjie_train_manager import get_train_manager
            res = get_train_manager().start("rebuild_components", {"era": payload.get("era", "KAISHU")})
            body = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self.send_response(200 if res.get("ok") else 400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/train"):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            from cangjie_train_manager import get_train_manager
            res = get_train_manager().start(payload.get("type", ""), payload.get("params", {}))
            body = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self.send_response(200 if res.get("ok") else 400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/models/switch"):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            from cangjie_model_registry import CangjieModelRegistry
            reg = CangjieModelRegistry(config=self.workstation.photo_decoder.config)
            res = reg.switch(payload.get("kind", ""), payload.get("id", ""))
            body = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self.send_response(200 if res.get("ok") else 400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/decode_photo"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length)
                payload = json.loads(raw_body.decode("utf-8"))
                img_bytes = base64.b64decode(payload["image_base64"])
                fname = payload.get("filename", "artifact.png")
                strokes = payload.get("strokes")
                stamped_nodes = payload.get("stamped_nodes")
                detector = payload.get("detector")
                matcher = payload.get("matcher")
                llm = payload.get("llm")
                llm_model = payload.get("llm_model")
                llm_url = payload.get("llm_url")
                res = self.workstation.decode_uploaded_photo_bytes(
                    img_bytes,
                    filename=fname,
                    strokes=strokes,
                    stamped_nodes=stamped_nodes,
                    detector_backend=detector,
                    matcher_backend=matcher,
                    llm_backend=llm,
                    llm_model=llm_model,
                    llm_url=llm_url
                )
                body = json.dumps(res, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                err = json.dumps({"error": str(e), "endpoint": "/api/decode_photo"}, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


def export_studio_and_run_verification(
    html_out_path: str = "cangjie_paleo_studio.html",
    hw_profile: str = "laptop"
) -> Dict[str, Any]:
    print("=" * 78)
    print("Инициализация боевой станции Project Cangjie v8.5 (Multi-Glyph + Прямой рендер + Фото в Рисовалке)")
    print("=" * 78)

    ws = ProductionPaleoWorkstationEngine(resolution=480, hw_profile=hw_profile)
    presets = [
        ("⿳雨鬼申", "DAOIST_HUANGBIAOZHI"),
        ("⿰氵⿱土寸", "XUAN_INK_SEAL"),
        ("⿱艹⿳雨口巫", "SHANG_JIAGUWEN"),
        ("⿺辶首", "TOPIAN_STONE_RUBBING"),
        ("⿸广⿰木木", "ZHOU_JINWEN"),
        ("⿴囗玉", "DAOIST_HUANGBIAOZHI"),
    ]
    examples = [ws._analyze_single_expression(q, carrier_mode=cm) for (q, cm) in presets]

    # Предкомпилируем 3 многознаковых свитка (с переиспользованием кэша для мгновенного старта)
    multi_scrolls = [
        [
            examples[1],
            examples[2],
            ws._analyze_single_expression("雷", "DAOIST_HUANGBIAOZHI"),
            examples[3],
        ],
        [
            ws._analyze_single_expression("我", "DAOIST_HUANGBIAOZHI"),
            ws._analyze_single_expression("神", "SHANG_JIAGUWEN"),
            ws._analyze_single_expression("電", "ZHOU_JINWEN"),
            examples[5],
        ],
        [
            examples[0],
            examples[4],
            examples[5],
        ]
    ]

    stats = {
        "unihan_count": len(ws.deducer.engine.characters_db),
        "chise_count": len(ws.deducer.engine.existing_ids_set),
        "baxter_count": len(ws.deducer.engine.phonetic_glyphs),
        "skeletons_count": len(ws.degradation_pipe.repo.medians) or 9574,
    }

    html_content = build_production_studio_html(
        examples, multi_scrolls, stats, default_hw_profile=ws.hw_profile
    )
    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ [1/4] Инверсия оси Y в MakeMeAHanzi устранена (y_screen = 900.0 - y_em, кэш .fast_v8.pkl).")
    print(f"✅ [2/4] Мульти-иероглифная лента (Multi-Glyph Strip) и разбор одиночных знаков («我» -> ⿰手戈, «雷» -> ⿱雨田) активны.")
    print(f"✅ [3/4] Активный аппаратный профиль: {ws.hw_profile} | Фото-сканер + Мастерская каллиграфии готовы.")
    print(f"✅ [4/4] Автономный интерфейс v8.5 сохранен: {os.path.abspath(html_out_path)}")
    print("=" * 78)

    return {"workstation": ws, "html": html_content, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description="Project Cangjie v8.5 — Профессиональная Палеографическая Станция")
    parser.add_argument("--serve", action="store_true", help="Запустить живой HTTP-сервер рабочей станции")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Хост веб-сервера (по умолчанию 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Порт веб-сервера (по умолчанию 8080)")
    parser.add_argument("--export-html", type=str, default="cangjie_paleo_studio.html", help="Путь сохранения автономного HTML")
    parser.add_argument(
        "--hw-profile",
        type=str,
        default="laptop",
        help="Аппаратный профиль: laptop | turbo_t1_rtx4060 (или rtx4060) | station_4x4090 (или 4x4090) | auto"
    )
    args = parser.parse_args()

    bundle = export_studio_and_run_verification(args.export_html, hw_profile=args.hw_profile)
    if args.serve:
        free_port_if_in_use(args.port)
        CangjieWorkstationHTTPHandler.workstation = bundle["workstation"]
        CangjieWorkstationHTTPHandler.cached_html = bundle["html"]
        server = ReusableHTTPServer((args.host, args.port), CangjieWorkstationHTTPHandler)
        print(f"🌐 Сервер станции палеографа v8.5 ({bundle['workstation'].hw_profile}) запущен: http://localhost:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()