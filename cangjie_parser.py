"""
Project Cangjie (Цанцзе) — Phase 1 Production Parser [ENTERPRISE GRADE v3.1]
Unihan & CHISE IDS Automated Parser, EASA Entropy Calculator & Knowledge Graph Generator.
Includes:
- Regex Tokenization for CHISE non-Unicode glyphs (&CDP-...;)
- Real EASA Shannon Self-Information Weighting: I(r) = -log2((freq(r)+1)/(N+214))
- Semantic ([:SEMANTIC]) vs Phonetic ([:PHONETIC]) Edge Role Resolution in AST
- Polyphone Baxter-Sagart Support & Collision-Free Dataset_C Synthesis
"""

from __future__ import annotations

import os
import re
import csv
import math
import json
import random
import logging
from typing import Dict, List, Any, Optional, Set, Iterator
from dataclasses import dataclass, asdict, field

try:
    from cangjie_benchmark_forge import (
        KANGXI_214_COMPREHENSIVE_CATALOG,
        PHONETICS_COMPREHENSIVE_DB,
        EXISTING_COMBINATIONS
    )
except ImportError:
    from cangjie_renderer import (
        KANGXI_214_COMPREHENSIVE_CATALOG,
        PHONETICS_COMPREHENSIVE_DB,
        EXISTING_COMBINATIONS
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PARSER]: %(message)s"
)
logger = logging.getLogger("CangjieProductionParser")

BINARY_OPERATORS: Set[str] = set("⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻")
TERNARY_OPERATORS: Set[str] = set("⿲⿳")
ALL_OPERATORS: Set[str] = BINARY_OPERATORS | TERNARY_OPERATORS
CHISE_TOKEN_REGEX = re.compile(r"&[^;]+;|.")


@dataclass
class KangxiRadical:
    radical_id: int
    is_simplified: bool
    glyph: str
    additional_strokes: int = 0
    corpus_frequency: int = 0
    information_content_bits: float = 0.0
    easa_weight: float = 1.0


@dataclass
class ASTNode:
    node_type: str  # 'operator' или 'component'
    value: str
    arity: int = 0
    children: List['ASTNode'] = field(default_factory=list)
    radical_info: Optional[KangxiRadical] = None
    role: Optional[str] = None       # 'semantic_radical', 'phonetic_component', 'structural_component'
    edge_type: Optional[str] = None  # 'SEMANTIC', 'PHONETIC', 'STRUCTURAL'
    position: Optional[str] = None
    easa_weight: float = 0.0


@dataclass
class CharacterEntity:
    codepoint: str
    glyph: str
    ids: str = ""
    ast: Optional[ASTNode] = None
    krs_unicode: List[KangxiRadical] = field(default_factory=list)
    baxter_sagart: Dict[str, Any] = field(default_factory=dict)
    easa_centroid_score: float = 0.0
    is_fake: bool = False


class IDSParser:
    def __init__(self, radical_map: Dict[str, KangxiRadical], phonetic_set: Set[str]):
        self.radical_map = radical_map
        self.phonetic_set = phonetic_set

    def parse_sequence(
        self,
        ids_sequence: str,
        primary_radical_id: Optional[int] = None
    ) -> Optional[ASTNode]:
        if not ids_sequence or not ids_sequence.strip():
            return None
        seq_parts = ids_sequence.split()
        if not seq_parts:
            return None
        clean_seq = seq_parts[0].strip()
        tokens = CHISE_TOKEN_REGEX.findall(clean_seq)
        if not tokens:
            return None

        iterator = iter(tokens)
        try:
            root = self._parse_node(iterator, position_context="root")
            self._annotate_ast_roles(root, primary_radical_id)
            return root
        except StopIteration:
            logger.debug(f"Неполная последовательность IDS: {ids_sequence}")
            return None
        except Exception as e:
            logger.error(f"Сбой парсинга IDS {ids_sequence}: {str(e)}")
            return None

    def _parse_node(self, iterator: Iterator[str], position_context: str) -> ASTNode:
        token = next(iterator)
        if token in ALL_OPERATORS:
            arity = 3 if token in TERNARY_OPERATORS else 2
            node = ASTNode(
                node_type="operator",
                value=token,
                arity=arity,
                position=position_context
            )
            child_positions = self._get_positional_context(token)
            for i in range(arity):
                child_pos = child_positions[i] if i < len(child_positions) else "nested"
                node.children.append(self._parse_node(iterator, child_pos))
            return node
        else:
            rad_info = self.radical_map.get(token)
            return ASTNode(
                node_type="component",
                value=token,
                radical_info=rad_info,
                role="structural_component",
                edge_type="STRUCTURAL",
                position=position_context,
                easa_weight=rad_info.easa_weight if rad_info else 0.5
            )

    def _annotate_ast_roles(self, node: ASTNode, primary_radical_id: Optional[int]) -> None:
        """Размечает [:SEMANTIC] и [:PHONETIC] ребра в AST на основе kRSUnicode и фонологического корпуса."""
        leaves: List[ASTNode] = []
        self._collect_leaves(node, leaves)
        if not leaves:
            return

        semantic_assigned = False
        # 1. Ищем точное совпадение с главным словарным радикалом из kRSUnicode
        if primary_radical_id is not None:
            for leaf in leaves:
                if leaf.radical_info and leaf.radical_info.radical_id == primary_radical_id:
                    leaf.role = "semantic_radical"
                    leaf.edge_type = "SEMANTIC"
                    semantic_assigned = True
                    break

        # 2. Если главный ключ не передан (например, в синтетике), берем первый известный радикал Канси в канонической позиции
        if not semantic_assigned:
            for leaf in leaves:
                if leaf.radical_info and leaf.position in {"left", "top", "surrounding", "root"}:
                    leaf.role = "semantic_radical"
                    leaf.edge_type = "SEMANTIC"
                    semantic_assigned = True
                    break

        # 3. Оставшиеся компоненты проверяем по корпусу фонетиков Baxter-Sagart
        for leaf in leaves:
            if leaf.edge_type == "SEMANTIC":
                continue
            if leaf.value in self.phonetic_set or len(leaves) == 2:
                leaf.role = "phonetic_component"
                leaf.edge_type = "PHONETIC"

    def _collect_leaves(self, node: ASTNode, acc: List[ASTNode]) -> None:
        if node.node_type == "component":
            acc.append(node)
        for ch in node.children:
            self._collect_leaves(ch, acc)

    @staticmethod
    def _get_positional_context(operator: str) -> List[str]:
        if operator == "⿰":
            return ["left", "right"]
        if operator == "⿱":
            return ["top", "bottom"]
        if operator == "⿲":
            return ["left", "middle", "right"]
        if operator == "⿳":
            return ["top", "middle", "bottom"]
        if operator in "⿴⿵⿶⿷⿸⿹⿺⿻":
            return ["surrounding", "enclosed"]
        return ["part_1", "part_2", "part_3"]


class ProjectCangjieEngine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.characters_db: Dict[str, CharacterEntity] = {}
        self.existing_ids_set: Set[str] = set()
        self.radical_id_to_info: Dict[int, KangxiRadical] = {}
        self.radical_glyph_to_info: Dict[str, KangxiRadical] = {}
        self.phonetic_glyphs: Set[str] = {item[0] for item in PHONETICS_COMPREHENSIVE_DB}
        self.krs_regex = re.compile(r"^(\d{1,3})('?)\.(-?\d+)$")

        for num, base_glyph, allograph, pinyin, strokes, meaning in KANGXI_214_COMPREHENSIVE_CATALOG:
            rad_obj = KangxiRadical(
                radical_id=num,
                is_simplified=False,
                glyph=base_glyph,
                additional_strokes=0
            )
            self.radical_id_to_info[num] = rad_obj
            self.radical_glyph_to_info[base_glyph] = rad_obj
            if allograph != base_glyph and allograph not in self.radical_glyph_to_info:
                self.radical_glyph_to_info[allograph] = rad_obj

    def load_unihan_radicals(self, filename: str = "Unihan_RadicalStrokes.txt") -> None:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            alt_path = os.path.join(self.data_dir, "Unihan_IRGSources.txt")
            if os.path.exists(alt_path):
                filepath = alt_path
            else:
                logger.warning(f"Файлы Unihan не найдены в {self.data_dir}. Используется встроенный реестр 214 ключей.")
                return

        logger.info(f"Парсинг {os.path.basename(filepath)}...")
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[1] == "kRSUnicode":
                    codepoint = parts[0]
                    krs_values = parts[2].split(" ")
                    try:
                        char_glyph = chr(int(codepoint.replace("U+", ""), 16))
                    except ValueError:
                        continue

                    radicals_for_char = []
                    for krs in krs_values:
                        match = self.krs_regex.match(krs)
                        if match:
                            rad_id = int(match.group(1))
                            is_simplified = (match.group(2) == "'")
                            add_strokes = int(match.group(3))

                            if rad_id in self.radical_id_to_info:
                                self.radical_id_to_info[rad_id].corpus_frequency += 1

                            rad_obj = KangxiRadical(
                                radical_id=rad_id,
                                is_simplified=is_simplified,
                                glyph=char_glyph,
                                additional_strokes=add_strokes
                            )
                            radicals_for_char.append(rad_obj)
                            if add_strokes == 0 and char_glyph not in self.radical_glyph_to_info:
                                self.radical_glyph_to_info[char_glyph] = self.radical_id_to_info.get(rad_id, rad_obj)

                    if radicals_for_char:
                        if char_glyph not in self.characters_db:
                            self.characters_db[char_glyph] = CharacterEntity(
                                codepoint=codepoint,
                                glyph=char_glyph,
                                ids=char_glyph
                            )
                        self.characters_db[char_glyph].krs_unicode = radicals_for_char
                        count += 1
        logger.info(f"Unihan: загружено {count} записей.")

    def load_baxter_sagart(self, filename: str = "Baxter_Sagart.csv") -> None:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Файл {filepath} не найден. Используется встроенный фонологический корпус.")
            return

        logger.info(f"Парсинг фонологических реконструкций Baxter-Sagart из {filename}...")
        matched = 0
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            field_map = {k.strip().lower(): k for k in (reader.fieldnames or [])}
            char_col = field_map.get("character") or field_map.get("char") or field_map.get("glyph") or field_map.get("word")
            py_col = field_map.get("pinyin") or field_map.get("py")
            mc_col = field_map.get("middle_chinese") or field_map.get("mc")
            oc_col = field_map.get("old_chinese") or field_map.get("oc")
            gsr_col = field_map.get("gsr")
            gloss_col = field_map.get("gloss") or field_map.get("meaning")
            # [PHONO-REPAIR v1.0] Колонка чтения ФОНЕТИКА серии: нужна, чтобы
            # для НЕСУЩЕСТВУЮЩИХ знаков выводить звучание по гнезду Сэшэн,
            # а для засвидетельствованных — собственное чтение знака.
            series_py_col = field_map.get("series_pinyin")
            bs_py_col = field_map.get("bs_pinyin")

            if not char_col:
                logger.error("В Baxter_Sagart.csv не найдена колонка знака.")
                return

            for row in reader:
                glyph = row.get(char_col, "").strip()
                if not glyph:
                    continue
                self.phonetic_glyphs.add(glyph)

                rec = {
                    "pinyin": row.get(py_col, "").strip() if py_col else "",
                    "bs_pinyin": row.get(bs_py_col, "").strip() if bs_py_col else "",
                    "series_pinyin": row.get(series_py_col, "").strip() if series_py_col else "",
                    "middle_chinese": row.get(mc_col, "").strip() if mc_col else "",
                    "old_chinese": row.get(oc_col, "").strip() if oc_col else "",
                    "gsr": row.get(gsr_col, "").strip() if gsr_col else "",
                    "gloss": row.get(gloss_col, "").strip() if gloss_col else ""
                }

                if glyph in self.characters_db:
                    char_entry = self.characters_db[glyph]
                    if not char_entry.baxter_sagart:
                        char_entry.baxter_sagart = {"primary": rec, "readings": [rec]}
                    else:
                        if "readings" not in char_entry.baxter_sagart:
                            char_entry.baxter_sagart["readings"] = [char_entry.baxter_sagart]
                        char_entry.baxter_sagart["readings"].append(rec)
                    matched += 1
                else:
                    codepoint = f"U+{ord(glyph[0]):04X}" if len(glyph) == 1 else "COMPLEX"
                    self.characters_db[glyph] = CharacterEntity(
                        codepoint=codepoint,
                        glyph=glyph,
                        ids=glyph,
                        baxter_sagart={"primary": rec, "readings": [rec]}
                    )
                    matched += 1

        logger.info(f"Baxter-Sagart: {matched} знаков интегрировано в онтологию.")

    def load_chise_ids(self, filename: str = "IDS-UCS-Basic.txt") -> None:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Файл {filepath} не найден. Пропуск CHISE IDS.")
            return

        logger.info(f"Построение AST-деревьев из {filename}...")
        parser = IDSParser(self.radical_glyph_to_info, self.phonetic_glyphs)
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                s_line = line.strip()
                if not s_line or s_line.startswith("#") or s_line.startswith(";"):
                    continue
                parts = [p.strip() for p in s_line.split("\t") if p.strip()]
                if len(parts) >= 3:
                    codepoint, glyph, raw_ids = parts[0], parts[1], parts[2]
                    raw_tokens = raw_ids.split()
                    if not raw_tokens:
                        continue
                    ids_seq = raw_tokens[0].strip()

                    primary_rad_id = None
                    if glyph in self.characters_db and self.characters_db[glyph].krs_unicode:
                        primary_rad_id = self.characters_db[glyph].krs_unicode[0].radical_id

                    ast_root = parser.parse_sequence(ids_seq, primary_radical_id=primary_rad_id)
                    self.existing_ids_set.add(ids_seq)

                    for token in CHISE_TOKEN_REGEX.findall(ids_seq):
                        if token in self.radical_glyph_to_info:
                            self.radical_glyph_to_info[token].corpus_frequency += 1

                    if glyph not in self.characters_db:
                        self.characters_db[glyph] = CharacterEntity(
                            codepoint=codepoint,
                            glyph=glyph,
                            ids=ids_seq
                        )
                    else:
                        self.characters_db[glyph].ids = ids_seq
                    self.characters_db[glyph].ast = ast_root
                    count += 1
        propagated = self.propagate_xiesheng_phonology()
        logger.info(
            f"Сгенерировано {count} деревьев AST CHISE. Индексировано {len(self.existing_ids_set)} IDS. "
            f"Распространено чтений по гнездам Сэшэн (谐声系): +{propagated} иероглифов."
        )

    def propagate_xiesheng_phonology(self) -> int:
        """
        Графовое распространение древнекитайских реконструкций Бакстера–Сагара по сериям Сэшэн (谐声系):
        Если составной иероглиф в CHISE IDS не имеет прямой записи в таблице Baxter_Sagart.csv,
        но содержит известный фонетический детерминатив (ребро [:PHONETIC] или правый/нижний/внутренний узел),
        он наследует архетип звучания OC/MC и код GSR своего фонетического ядра.
        """
        propagated_count = 0
        for glyph, ent in self.characters_db.items():
            if ent.baxter_sagart or not ent.ast:
                continue
            leaves: List[ASTNode] = []
            self._collect_ast_leaves(ent.ast, leaves)
            if len(leaves) < 2:
                continue
            # Ищем фонетический узел (сначала помеченный как PHONETIC, затем правый/нижний/внутренний)
            phon_candidates = [lf.value for lf in leaves if lf.edge_type == "PHONETIC"] + [lf.value for lf in reversed(leaves)]
            for cand in phon_candidates:
                cand_ent = self.characters_db.get(cand)
                if cand_ent and cand_ent.baxter_sagart and cand_ent.baxter_sagart.get("primary"):
                    src_rec = cand_ent.baxter_sagart["primary"]
                    # [PHONO-REPAIR v1.0] Чтение САМОГО знака НЕ наследуется
                    # от фонетика: 形声 даёт знаку СЕМЬЮ по фонетику, но чтение
                    # знак наследует лишь ЧАСТИЧНО (河 hé ≠ 可 kě). Наследуются
                    # только архетип серии (MC/OC) и код ГСР; `pinyin` берётся
                    # из Unihan/Бакстера-Сагарта, а если его нет — остаётся
                    # пустым, и движок честно помечает чтение как неизвестное.
                    inherited = {
                        "pinyin": "",
                        "own_pinyin": "",
                        "series_pinyin": src_rec.get("pinyin", ""),
                        "middle_chinese": src_rec.get("middle_chinese", ""),
                        "old_chinese": src_rec.get("old_chinese", ""),
                        "gsr": f"{src_rec.get('gsr', 'XIESHENG')}->{cand}",
                        "gloss": f"серия Сэшэн от фонетика «{cand}» ({src_rec.get('gloss', '')})"
                    }
                    ent.baxter_sagart = {"primary": inherited, "readings": [inherited]}
                    propagated_count += 1
                    break
        return propagated_count

    def _collect_ast_leaves(self, node: ASTNode, acc: List[ASTNode]) -> None:
        if node.node_type == "component":
            acc.append(node)
        for ch in node.children:
            self._collect_ast_leaves(ch, acc)

    def compute_easa_entropy_weights(self) -> None:
        """
        Алгоритм EASA (Entropy-Aware Structural Alignment):
        Вычисляет количество собственной информации Шеннона I(r) = -log2((freq(r) + 1) / (N_total + 214))
        для каждого из 214 радикалов Канси и обновляет центроиды иероглифов.
        """
        total_occurrences = sum(r.corpus_frequency for r in self.radical_id_to_info.values())
        denom = float(total_occurrences + 214)

        max_info = math.log2(denom) if denom > 1.0 else 1.0
        for rad in self.radical_id_to_info.values():
            prob = (rad.corpus_frequency + 1.0) / denom
            info_bits = -math.log2(prob)
            rad.information_content_bits = round(info_bits, 4)
            rad.easa_weight = round(info_bits / max_info, 4)

        for entity in self.characters_db.values():
            if entity.ast:
                weights: List[float] = []
                self._update_ast_easa(entity.ast, weights)
                if weights:
                    entity.easa_centroid_score = round(sum(weights) / len(weights), 4)
            elif entity.krs_unicode:
                rid = entity.krs_unicode[0].radical_id
                if rid in self.radical_id_to_info:
                    entity.easa_centroid_score = self.radical_id_to_info[rid].easa_weight

        logger.info("EASA: Информационная энтропия Шеннона рассчитана для всех 214 радикалов и узлов AST.")

    def _update_ast_easa(self, node: ASTNode, acc: List[float]) -> None:
        if node.node_type == "component" and node.radical_info:
            canonical = self.radical_id_to_info.get(node.radical_info.radical_id)
            if canonical:
                node.radical_info.corpus_frequency = canonical.corpus_frequency
                node.radical_info.information_content_bits = canonical.information_content_bits
                node.radical_info.easa_weight = canonical.easa_weight
                node.easa_weight = canonical.easa_weight
                acc.append(canonical.easa_weight)
        for ch in node.children:
            self._update_ast_easa(ch, acc)

    def generate_dataset_c(self, num_samples: int = 50) -> List[CharacterEntity]:
        logger.info(f"Генерация Dataset_C ({num_samples} выдуманных иероглифов)...")
        dataset_c: List[CharacterEntity] = []
        seen_fake: Set[str] = set()

        available_radicals = [r[1] for r in KANGXI_214_COMPREHENSIVE_CATALOG]
        parser = IDSParser(self.radical_glyph_to_info, self.phonetic_glyphs)
        attempts = 0
        max_attempts = num_samples * 200

        while len(dataset_c) < num_samples and attempts < max_attempts:
            attempts += 1
            op = random.choice(["⿰", "⿱"])
            rad1 = random.choice(available_radicals)
            rad2 = random.choice(available_radicals)
            fake_ids_seq = f"{op}{rad1}{rad2}"

            if (
                fake_ids_seq in seen_fake
                or fake_ids_seq in self.existing_ids_set
                or fake_ids_seq in self.characters_db
                or (op, rad1, rad2) in EXISTING_COMBINATIONS
            ):
                continue

            seen_fake.add(fake_ids_seq)
            fake_ast = parser.parse_sequence(fake_ids_seq)
            weights: List[float] = []
            if fake_ast:
                self._update_ast_easa(fake_ast, weights)

            fake_char = CharacterEntity(
                codepoint=f"FAKE_U+{len(dataset_c):04X}",
                glyph=fake_ids_seq,
                ids=fake_ids_seq,
                ast=fake_ast,
                easa_centroid_score=round(sum(weights) / len(weights), 4) if weights else 0.5,
                is_fake=True
            )
            dataset_c.append(fake_char)

        logger.info(f"Dataset_C сформирован: {len(dataset_c)} верифицированных псевдоиероглифов.")
        return dataset_c

    def export_neo4j_ontology(self, output_file: str) -> None:
        self.compute_easa_entropy_weights()
        logger.info(f"Экспорт онтологии в {output_file}...")
        fake_chars = self.generate_dataset_c(50)
        export_data = [asdict(c) for c in self.characters_db.values()]
        export_data.extend([asdict(fc) for fc in fake_chars])

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Экспорт завершен: {len(export_data)} узлов выгружено.")


def main():
    data_dir = os.path.abspath(os.getenv("CANGJIE_DATA_DIR", "./data"))
    output_file = os.path.join(data_dir, "neo4j_ontology_dump.json")
    os.makedirs(data_dir, exist_ok=True)

    engine = ProjectCangjieEngine(data_dir=data_dir)
    engine.load_unihan_radicals("Unihan_RadicalStrokes.txt")
    engine.load_baxter_sagart("Baxter_Sagart.csv")
    engine.load_chise_ids("IDS-UCS-Basic.txt")
    engine.export_neo4j_ontology(output_file)


if __name__ == "__main__":
    main()