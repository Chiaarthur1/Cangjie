"""
CANGJIE-CORE v3.1 — Robust CJK Neologism Synthesis & Vector Rendering Engine
Full 214 Kangxi Radicals, Phonetics Corpus, Zero-Cross Guarantee & Safe Allographs.
"""

from __future__ import annotations

import os
import sys
import json
import math
import logging
import subprocess
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, Any
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [RENDERER]: %(message)s")
logger = logging.getLogger("CangjieRenderer")


class SpatialSlot(Enum):
    ROOT = auto()
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()
    SURROUND_BL = auto()
    INNER = auto()


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def w(self) -> float:
        return self.width

    @property
    def h(self) -> float:
        return self.height

    @property
    def x1(self) -> float:
        return self.x

    @property
    def y1(self) -> float:
        return self.y

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def xc(self) -> float:
        return self.x + 0.5 * self.width

    @property
    def yc(self) -> float:
        return self.y + 0.5 * self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    def inset(self, dx: float, dy: float) -> BoundingBox:
        return BoundingBox(
            x=self.x + dx,
            y=self.y + dy,
            width=max(1.0, self.width - 2.0 * dx),
            height=max(1.0, self.height - 2.0 * dy)
        )


# ==============================================================================
# ПОЛНЫЙ КАНОНИЧЕСКИЙ РЕЕСТР 214 КЛЮЧЕЙ КАНСИ (1..214)
# ==============================================================================
KANGXI_214_COMPREHENSIVE_CATALOG: List[Tuple[int, str, str, str, int, str]] = [
    (1, "一", "一", "yī", 1, "one"), (2, "丨", "丨", "gǔn", 1, "line"), (3, "丶", "丶", "zhǔ", 1, "dot"),
    (4, "丿", "丿", "piě", 1, "slash"), (5, "乙", "乙", "yǐ", 1, "second"), (6, "亅", "亅", "jué", 1, "hook"),
    (7, "二", "二", "èr", 2, "two"), (8, "亠", "亠", "tóu", 2, "lid"), (9, "人", "亻", "rén", 2, "human"),
    (10, "儿", "儿", "ér", 2, "legs"), (11, "入", "入", "rù", 2, "enter"), (12, "八", "八", "bā", 2, "eight"),
    (13, "冂", "冂", "jiōng", 2, "down_box"), (14, "冖", "冖", "mì", 2, "cover"), (15, "冫", "冫", "bīng", 2, "ice"),
    (16, "几", "几", "jī", 2, "table"), (17, "凵", "凵", "kǎn", 2, "container"), (18, "刀", "刂", "dāo", 2, "knife"),
    (19, "力", "力", "lì", 2, "power"), (20, "勹", "勹", "bāo", 2, "wrap"), (21, "匕", "匕", "bǐ", 2, "spoon"),
    (22, "匚", "匚", "fāng", 2, "box"), (23, "匸", "匸", "xǐ", 2, "hiding_enclosure"), (24, "十", "十", "shí", 2, "ten"),
    (25, "卜", "卜", "bǔ", 2, "divination"), (26, "卩", "卩", "jié", 2, "seal"), (27, "厂", "厂", "hǎn", 2, "cliff"),
    (28, "厶", "厶", "sī", 2, "private"), (29, "又", "又", "yòu", 2, "again_right_hand"),
    (30, "口", "口", "kǒu", 3, "mouth"), (31, "囗", "囗", "wéi", 3, "enclosure"), (32, "土", "土", "tǔ", 3, "earth"),
    (33, "士", "士", "shì", 3, "scholar"), (34, "夂", "夂", "zhǐ", 3, "go"), (35, "夊", "夊", "suī", 3, "slow_go"),
    (36, "夕", "夕", "xī", 3, "evening"), (37, "大", "大", "dà", 3, "big"), (38, "女", "女", "nǚ", 3, "woman"),
    (39, "子", "子", "zǐ", 3, "child"), (40, "宀", "宀", "mián", 3, "roof"), (41, "寸", "寸", "cùn", 3, "inch"),
    (42, "小", "小", "xiǎo", 3, "small"), (43, "尢", "尢", "wāng", 3, "lame"), (44, "尸", "尸", "shī", 3, "corpse"),
    (45, "屮", "屮", "chè", 3, "sprout"), (46, "山", "山", "shān", 3, "mountain"), (47, "巛", "巛", "chuān", 3, "river"),
    (48, "工", "工", "gōng", 3, "work"), (49, "己", "己", "jǐ", 3, "oneself"), (50, "巾", "巾", "jīn", 3, "turban"),
    (51, "干", "干", "gān", 3, "dry"), (52, "幺", "幺", "yāo", 3, "short"), (53, "广", "广", "guǎng", 3, "broad_cliff"),
    (54, "廴", "廴", "yǐn", 3, "long_stride"), (55, "廾", "廾", "gǒng", 3, "two_hands"), (56, "弋", "弋", "yì", 3, "shoot"),
    (57, "弓", "弓", "gōng", 3, "bow"), (58, "彐", "彐", "jì", 3, "snout"), (59, "彡", "彡", "shān", 3, "hair"),
    (60, "彳", "彳", "chì", 3, "step"), (61, "心", "忄", "xīn", 4, "heart"), (62, "戈", "戈", "gē", 4, "spear"),
    (63, "戶", "戶", "hù", 4, "door"), (64, "手", "扌", "shǒu", 4, "hand"), (65, "支", "支", "zhī", 4, "branch"),
    (66, "攴", "攵", "pū", 4, "rap"), (67, "文", "文", "wén", 4, "script"), (68, "斗", "斗", "dǒu", 4, "dipper"),
    (69, "斤", "斤", "jīn", 4, "axe"), (70, "方", "方", "fāng", 4, "square"), (71, "无", "无", "wú", 4, "not"),
    (72, "日", "日", "rì", 4, "sun"), (73, "曰", "曰", "yuē", 4, "say"), (74, "月", "月", "yuè", 4, "moon"),
    (75, "木", "朩", "mù", 4, "tree"), (76, "欠", "欠", "qiàn", 4, "yawn"), (77, "止", "止", "zhǐ", 4, "stop"),
    (78, "歹", "歹", "dǎi", 4, "death"), (79, "殳", "殳", "shū", 4, "weapon"), (80, "毋", "毋", "wú", 4, "do_not"),
    (81, "比", "比", "bǐ", 4, "compare"), (82, "毛", "毛", "máo", 4, "fur"), (83, "氏", "氏", "shì", 4, "clan"),
    (84, "气", "气", "qì", 4, "steam_qi"), (85, "水", "氵", "shuǐ", 4, "water"), (86, "火", "灬", "huǒ", 4, "fire"),
    (87, "爪", "爫", "zhǎo", 4, "claw"), (88, "父", "父", "fù", 4, "father"), (89, "爻", "爻", "yáo", 4, "lines"),
    (90, "爿", "爿", "qiáng", 4, "split_wood"), (91, "片", "片", "piàn", 4, "slice"), (92, "牙", "牙", "yá", 4, "fang"),
    (93, "牛", "牜", "niú", 4, "cow"), (94, "犬", "犭", "quǎn", 4, "dog"),
    (95, "玄", "玄", "xuán", 5, "profound"), (96, "玉", "王", "yù", 5, "jade"), (97, "瓜", "瓜", "guā", 5, "melon"),
    (98, "瓦", "瓦", "wǎ", 5, "tile"), (99, "甘", "甘", "gān", 5, "sweet"), (100, "生", "生", "shēng", 5, "life"),
    (101, "用", "用", "yòng", 5, "use"), (102, "田", "田", "tián", 5, "field"), (103, "疋", "疋", "pǐ", 5, "bolt_of_cloth"),
    (104, "疒", "疒", "nè", 5, "sickness"), (105, "癶", "癶", "bō", 5, "footsteps"), (106, "白", "白", "bái", 5, "white"),
    (107, "皮", "皮", "pí", 5, "skin"), (108, "皿", "皿", "mǐn", 5, "dish"), (109, "目", "目", "mù", 5, "eye"),
    (110, "矛", "矛", "máo", 5, "spear"), (111, "矢", "矢", "shǐ", 5, "arrow"), (112, "石", "石", "shí", 5, "stone"),
    (113, "示", "礻", "shì", 5, "spirit_ancestor"), (114, "禸", "禸", "róu", 5, "track"), (115, "禾", "禾", "hé", 5, "grain"),
    (116, "穴", "穴", "xué", 5, "cave"), (117, "立", "立", "lì", 5, "stand"),
    (118, "竹", "⺮", "zhú", 6, "bamboo"), (119, "米", "米", "mǐ", 6, "rice"), (120, "糸", "纟", "mì", 6, "silk"),
    (121, "缶", "缶", "fǒu", 6, "jar"), (122, "网", "罒", "wǎng", 6, "net"), (123, "羊", "⺷", "yáng", 6, "sheep"),
    (124, "羽", "羽", "yǔ", 6, "feather"), (125, "老", "耂", "lǎo", 6, "old"), (126, "而", "而", "ér", 6, "and"),
    (127, "耒", "耒", "lěi", 6, "plow"), (128, "耳", "耳", "ěr", 6, "ear"), (129, "聿", "聿", "yù", 6, "brush"),
    (130, "肉", "月", "ròu", 6, "meat_flesh"), (131, "臣", "臣", "chén", 6, "minister"), (132, "自", "自", "zì", 6, "oneself"),
    (133, "至", "至", "zhì", 6, "arrive"), (134, "臼", "臼", "jiù", 6, "mortar"), (135, "舌", "舌", "shé", 6, "tongue"),
    (136, "舛", "舛", "chuǎn", 6, "oppose"), (137, "舟", "舟", "zhōu", 6, "boat"), (138, "艮", "艮", "gèn", 6, "stopping"),
    (139, "色", "色", "sè", 6, "color"), (140, "艸", "艹", "cǎo", 6, "grass"), (141, "虍", "虍", "hū", 6, "tiger"),
    (142, "虫", "虫", "chóng", 6, "insect"), (143, "血", "血", "xuè", 6, "blood"), (144, "行", "行", "xíng", 6, "walk"),
    (145, "衣", "衤", "yī", 6, "clothes"), (146, "襾", "覀", "yà", 6, "west_cover"),
    (147, "見", "见", "jiàn", 7, "see"), (148, "角", "角", "jiǎo", 7, "horn"), (149, "言", "讠", "yán", 7, "speech"),
    (150, "谷", "谷", "gǔ", 7, "valley"), (151, "豆", "豆", "dòu", 7, "bean"), (152, "豕", "豕", "shǐ", 7, "pig"),
    (153, "豸", "豸", "zhì", 7, "badger"), (154, "貝", "贝", "bèi", 7, "shell_money"), (155, "赤", "赤", "chì", 7, "red"),
    (156, "走", "走", "zǒu", 7, "run"), (157, "足", "⻊", "zú", 7, "foot"), (158, "身", "身", "shēn", 7, "body"),
    (159, "車", "车", "chē", 7, "cart"), (160, "辛", "辛", "xīn", 7, "bitter"), (161, "辰", "辰", "chén", 7, "morning"),
    (162, "辵", "辶", "chuò", 7, "walk_stride"), (163, "邑", "阝", "yì", 7, "city_right"), (164, "酉", "酉", "yǒu", 7, "wine"),
    (165, "釆", "釆", "biàn", 7, "distinguish"), (166, "里", "里", "lǐ", 7, "village"),
    (167, "金", "钅", "jīn", 8, "metal"), (168, "長", "长", "cháng", 8, "long"), (169, "門", "门", "mén", 8, "gate"),
    (170, "阜", "阝", "fù", 8, "mound_left"), (171, "隶", "隶", "lì", 8, "slave"), (172, "隹", "隹", "zhuī", 8, "short_tailed_bird"),
    (173, "雨", "⻗", "yǔ", 8, "rain"), (174, "靑", "青", "qīng", 8, "blue_green"), (175, "非", "非", "fēi", 8, "wrong"),
    (176, "面", "面", "miàn", 9, "face"), (177, "革", "革", "gé", 9, "leather"), (178, "韋", "韦", "wéi", 9, "tanned_leather"),
    (179, "韭", "韭", "jiǔ", 9, "leek"), (180, "音", "音", "yīn", 9, "sound"), (181, "頁", "页", "yè", 9, "leaf_head"),
    (182, "風", "风", "fēng", 9, "wind"), (183, "飛", "飞", "fēi", 9, "fly"), (184, "食", "饣", "shí", 9, "food"),
    (185, "首", "首", "shǒu", 9, "head"), (186, "香", "香", "xiāng", 9, "fragrant"),
    (187, "馬", "马", "mǎ", 10, "horse"), (188, "骨", "骨", "gǔ", 10, "bone"), (189, "高", "高", "gāo", 10, "tall"),
    (190, "髟", "髟", "biāo", 10, "long_hair"), (191, "鬥", "鬥", "dòu", 10, "fight"), (192, "鬯", "鬯", "chàng", 10, "sacrificial_wine"),
    (193, "鬲", "鬲", "lì", 10, "cauldron"), (194, "鬼", "鬼", "guǐ", 10, "ghost"),
    (195, "魚", "鱼", "yú", 11, "fish"), (196, "鳥", "鸟", "niǎo", 11, "bird"), (197, "鹵", "卤", "lǔ", 11, "salt"),
    (198, "鹿", "鹿", "lù", 11, "deer"), (199, "麥", "麦", "mài", 11, "wheat"), (200, "麻", "麻", "má", 11, "hemp"),
    (201, "黃", "黄", "huáng", 12, "yellow"), (202, "黍", "黍", "shǔ", 12, "millet"), (203, "黑", "黑", "hēi", 12, "black"),
    (204, "黹", "黹", "zhǐ", 12, "embroidery"),
    (205, "黽", "黾", "mǐn", 13, "frog"), (206, "鼎", "鼎", "dǐng", 13, "tripod"), (207, "鼓", "鼓", "gǔ", 13, "drum"),
    (208, "鼠", "鼠", "shǔ", 13, "rat"),
    (209, "鼻", "鼻", "bí", 14, "nose"), (210, "齊", "齐", "qí", 14, "even"),
    (211, "齒", "齿", "chǐ", 15, "tooth"),
    (212, "龍", "龙", "lóng", 16, "dragon"), (213, "龜", "龟", "guī", 16, "turtle"),
    (214, "龠", "龠", "yuè", 17, "flute")
]


# ==============================================================================
# ПРОДУКТИВНЫЙ ФОНЕТИЧЕСКИЙ КОРПУС ДРЕВНЕКИТАЙСКОГО ЯЗЫКА
# ==============================================================================
PHONETICS_COMPREHENSIVE_DB: List[Tuple[str, str, int]] = [
    ("寺", "sì", 6), ("青", "qīng", 8), ("尧", "yáo", 6), ("仓", "cāng", 4),
    ("兆", "zhào", 6), ("甫", "fǔ", 7), ("龠", "yuè", 17), ("包", "bāo", 5),
    ("戋", "jiān", 5), ("莫", "mò", 10), ("辟", "bì", 13), ("卑", "bēi", 8),
    ("畐", "fú", 9), ("圭", "guī", 6), ("宛", "wǎn", 8), ("艮", "gèn", 6),
    ("俞", "yú", 9), ("扁", "biǎn", 9), ("甬", "yǒng", 7), ("仑", "lún", 4),
    ("登", "dēng", 12), ("占", "zhān", 5), ("兼", "jiān", 10), ("咸", "xián", 9),
    ("垂", "chuí", 8), ("采", "cǎi", 8), ("巠", "jīng", 7), ("召", "zhào", 5),
    ("喿", "zào", 13), ("夋", "qūn", 7), ("夬", "guài", 4), ("亢", "kàng", 4),
    ("奇", "qí", 8), ("乔", "qiáo", 6), ("尚", "shàng", 8), ("肖", "xiào", 7),
    ("夆", "féng", 7), ("孚", "fú", 7), ("粦", "lín", 12), ("曼", "màn", 11),
    ("奉", "fèng", 8), ("炎", "yán", 8), ("戉", "yuè", 5), ("殳", "shū", 4),
    ("乍", "zhà", 5), ("生", "shēng", 5), ("丁", "dīng", 2), ("巴", "bā", 4),
    ("凡", "fán", 3), ("反", "fǎn", 4), ("中", "zhōng", 4), ("方", "fāng", 4),
    ("白", "bái", 5), ("百", "bǎi", 6), ("且", "qiě", 5), ("旦", "dàn", 5),
    ("丙", "bǐng", 5), ("甲", "jiǎ", 5), ("申", "shēn", 5), ("由", "yóu", 5)
]


class SafeAllographResolver:
    """
    Разрешает позиционные трансформации ключей с разделением топологии.
    Особое внимание уделено двуликому 阝 (170 阜 vs 163 邑) и 肉月旁.
    """
    PRIMARY_VARIANTS: Dict[str, Dict[SpatialSlot, str]] = {
        "水": {SpatialSlot.LEFT: "氵", SpatialSlot.BOTTOM: "氺"},
        "人": {SpatialSlot.LEFT: "亻", SpatialSlot.TOP: "𠆢"},
        "手": {SpatialSlot.LEFT: "扌"},
        "心": {SpatialSlot.LEFT: "忄", SpatialSlot.BOTTOM: "灬"},
        "火": {SpatialSlot.BOTTOM: "灬"},
        "刀": {SpatialSlot.RIGHT: "刂"},
        "犬": {SpatialSlot.LEFT: "犭"},
        "示": {SpatialSlot.LEFT: "礻"},
        "衣": {SpatialSlot.LEFT: "衤"},
        "肉": {SpatialSlot.LEFT: "月"},  # 肉月旁
        "竹": {SpatialSlot.TOP: "⺮"},
        "艸": {SpatialSlot.TOP: "艹"},
        "网": {SpatialSlot.TOP: "罒"},
        "羊": {SpatialSlot.TOP: "⺷"},
        "金": {SpatialSlot.LEFT: "钅"},
        "言": {SpatialSlot.LEFT: "讠"},
        "食": {SpatialSlot.LEFT: "饣"},
        "糸": {SpatialSlot.LEFT: "纟"},
        "車": {SpatialSlot.LEFT: "车"},
        "邑": {SpatialSlot.RIGHT: "阝"},  # 右耳旁 (город)
        "阜": {SpatialSlot.LEFT: "阝"},   # 左耳旁 (холм)
        "木": {SpatialSlot.LEFT: "朩"},
        "辵": {SpatialSlot.SURROUND_BL: "辶"},
        "足": {SpatialSlot.LEFT: "⻊"},
        "雨": {SpatialSlot.TOP: "⻗"},
        "玉": {SpatialSlot.LEFT: "王"},
        "牛": {SpatialSlot.LEFT: "牜"},
        "老": {SpatialSlot.TOP: "耂"},
        "攴": {SpatialSlot.RIGHT: "攵"},
        "疋": {SpatialSlot.TOP: "⺪"}
    }

    TRAD_SIMP_FALLBACK: Dict[str, str] = {
        "車": "车", "門": "门", "見": "见", "言": "讠", "貝": "贝",
        "金": "钅", "長": "长", "食": "饣", "馬": "马", "魚": "鱼",
        "鳥": "鸟", "齊": "齐", "齒": "齿", "龍": "龙", "龜": "龟"
    }

    @classmethod
    def resolve(cls, radical: str, slot: SpatialSlot, available_chars: Set[str]) -> str:
        target = cls.PRIMARY_VARIANTS.get(radical, {}).get(slot, radical)
        if target in available_chars:
            return target
        alt = cls.TRAD_SIMP_FALLBACK.get(target, cls.TRAD_SIMP_FALLBACK.get(radical))
        if alt and alt in available_chars:
            return alt
        if radical in available_chars:
            return radical
        return target


class GlyphRepository:
    """Репозиторий медианных штрихов с авто-подключением MakeMeAHanzi и шрифтового fallback."""
    def __init__(self, graphics_path: Optional[str] = None):
        self.medians: Dict[str, List[List[Tuple[float, float]]]] = {}
        self.stroke_counts: Dict[str, int] = {}
        self.content_bboxes: Dict[str, BoundingBox] = {}

        for _, base_g, allo_g, _, strokes, _ in KANGXI_214_COMPREHENSIVE_CATALOG:
            self.stroke_counts[base_g] = strokes
            self.stroke_counts[allo_g] = strokes
        for phon_g, _, strokes in PHONETICS_COMPREHENSIVE_DB:
            self.stroke_counts[phon_g] = strokes

        self._init_builtin_medians()

        candidate_paths = [
            graphics_path,
            os.getenv("CANGJIE_GRAPHICS_PATH"),
            os.path.join(os.getenv("CANGJIE_DATA_DIR", "./data"), "graphics.txt"),
            "./data/graphics.txt"
        ]
        for p in candidate_paths:
            if p and os.path.exists(p):
                self._load_external_graphics(p)
                break

    def _init_builtin_medians(self) -> None:
        raw_seed = {
            "氵": [[(280.0, 240.0), (370.0, 310.0)], [(250.0, 470.0), (340.0, 520.0)], [(220.0, 800.0), (380.0, 680.0)]],
            "木": [[(180.0, 400.0), (840.0, 400.0)], [(512.0, 160.0), (512.0, 880.0)], [(512.0, 400.0), (280.0, 760.0)], [(512.0, 400.0), (760.0, 760.0)]],
            "朩": [[(180.0, 400.0), (760.0, 400.0)], [(460.0, 160.0), (460.0, 880.0)], [(460.0, 400.0), (220.0, 760.0)], [(460.0, 500.0), (600.0, 640.0)]],
            "亻": [[(560.0, 160.0), (280.0, 520.0)], [(420.0, 380.0), (420.0, 880.0)]],
            "扌": [[(200.0, 420.0), (720.0, 380.0)], [(480.0, 160.0), (480.0, 880.0), (380.0, 820.0)], [(220.0, 680.0), (640.0, 540.0)]],
            "忄": [[(300.0, 400.0), (400.0, 480.0)], [(512.0, 160.0), (512.0, 880.0)], [(700.0, 360.0), (600.0, 480.0)]],
            "艹": [[(160.0, 500.0), (860.0, 500.0)], [(360.0, 240.0), (360.0, 760.0)], [(660.0, 240.0), (660.0, 760.0)]],
            "钅": [[(512.0, 160.0), (360.0, 340.0)], [(420.0, 280.0), (640.0, 360.0)], [(260.0, 460.0), (680.0, 440.0)], [(240.0, 600.0), (700.0, 580.0)], [(460.0, 460.0), (460.0, 860.0)], [(220.0, 840.0), (760.0, 720.0)]],
            "讠": [[(420.0, 240.0), (560.0, 340.0)], [(300.0, 480.0), (640.0, 480.0), (480.0, 780.0), (360.0, 740.0)]],
            "犭": [[(620.0, 180.0), (320.0, 480.0)], [(380.0, 300.0), (520.0, 520.0), (380.0, 880.0), (280.0, 800.0)], [(240.0, 620.0), (640.0, 420.0)]],
            "火": [[(320.0, 400.0), (240.0, 560.0)], [(700.0, 360.0), (780.0, 540.0)], [(512.0, 160.0), (340.0, 860.0)], [(512.0, 360.0), (820.0, 860.0)]],
            "灬": [[(220.0, 480.0), (160.0, 640.0)], [(400.0, 480.0), (440.0, 640.0)], [(600.0, 480.0), (640.0, 640.0)], [(800.0, 480.0), (860.0, 640.0)]],
            "日": [[(280.0, 200.0), (280.0, 820.0)], [(280.0, 200.0), (740.0, 200.0), (740.0, 820.0)], [(280.0, 500.0), (740.0, 500.0)], [(280.0, 820.0), (740.0, 820.0)]],
            "月": [[(320.0, 180.0), (280.0, 860.0)], [(320.0, 180.0), (720.0, 180.0), (720.0, 860.0), (620.0, 820.0)], [(320.0, 400.0), (720.0, 400.0)], [(320.0, 620.0), (720.0, 620.0)]],
            "土": [[(240.0, 460.0), (784.0, 460.0)], [(512.0, 160.0), (512.0, 820.0)], [(160.0, 820.0), (864.0, 820.0)]],
            "女": [[(512.0, 160.0), (360.0, 520.0), (620.0, 780.0)], [(360.0, 520.0), (280.0, 880.0)], [(160.0, 500.0), (864.0, 500.0)]],
            "口": [[(260.0, 240.0), (260.0, 800.0)], [(260.0, 240.0), (764.0, 240.0), (764.0, 800.0)], [(260.0, 800.0), (764.0, 800.0)]],
            "山": [[(512.0, 180.0), (512.0, 840.0)], [(240.0, 420.0), (240.0, 840.0), (784.0, 840.0)], [(784.0, 420.0), (784.0, 840.0)]],
            "寺": [[(260.0, 220.0), (760.0, 220.0)], [(512.0, 120.0), (512.0, 420.0)], [(160.0, 420.0), (860.0, 420.0)], [(320.0, 560.0), (320.0, 880.0)], [(320.0, 560.0), (720.0, 560.0), (720.0, 920.0), (620.0, 900.0)], [(512.0, 640.0), (420.0, 760.0)]],
            "青": [[(280.0, 200.0), (740.0, 200.0)], [(512.0, 120.0), (512.0, 480.0)], [(340.0, 320.0), (680.0, 320.0)], [(200.0, 480.0), (820.0, 480.0)], [(320.0, 540.0), (320.0, 920.0)], [(320.0, 540.0), (700.0, 540.0), (700.0, 920.0)], [(320.0, 680.0), (700.0, 680.0)], [(320.0, 800.0), (700.0, 800.0)]],
            "尧": [[(280.0, 220.0), (740.0, 220.0)], [(512.0, 140.0), (512.0, 360.0)], [(220.0, 360.0), (800.0, 360.0)], [(400.0, 460.0), (320.0, 880.0)], [(560.0, 460.0), (640.0, 760.0), (760.0, 820.0)]],
            "仓": [[(512.0, 140.0), (200.0, 540.0)], [(512.0, 140.0), (820.0, 540.0)], [(320.0, 600.0), (320.0, 880.0)], [(320.0, 600.0), (700.0, 600.0), (700.0, 880.0), (580.0, 840.0)]],
            "兆": [[(380.0, 180.0), (380.0, 840.0)], [(380.0, 380.0), (200.0, 580.0)], [(380.0, 600.0), (180.0, 800.0)], [(640.0, 180.0), (640.0, 840.0), (760.0, 760.0)], [(640.0, 380.0), (820.0, 580.0)], [(640.0, 600.0), (840.0, 800.0)]],
            "甫": [[(512.0, 120.0), (512.0, 320.0)], [(220.0, 320.0), (804.0, 320.0)], [(320.0, 320.0), (320.0, 880.0)], [(320.0, 320.0), (720.0, 320.0), (720.0, 880.0)], [(320.0, 520.0), (720.0, 520.0)], [(320.0, 700.0), (720.0, 700.0)], [(512.0, 320.0), (512.0, 920.0)]],
            "龠": [[(400.0, 160.0), (624.0, 160.0)], [(512.0, 160.0), (512.0, 360.0)], [(260.0, 360.0), (764.0, 360.0)], [(320.0, 420.0), (320.0, 880.0)], [(512.0, 420.0), (512.0, 880.0)], [(704.0, 420.0), (704.0, 880.0)], [(260.0, 880.0), (764.0, 880.0)]]
        }
        for char, strokes in raw_seed.items():
            self._register_glyph(char, strokes)

    _GLOBAL_SKELETON_CACHE: Optional[Tuple[Dict[str, Any], Dict[str, int], Dict[str, BoundingBox]]] = None

    def _load_external_graphics(self, path: str) -> None:
        if GlyphRepository._GLOBAL_SKELETON_CACHE is not None:
            m_c, s_c, b_c = GlyphRepository._GLOBAL_SKELETON_CACHE
            self.medians.update(m_c)
            self.stroke_counts.update(s_c)
            self.content_bboxes.update(b_c)
            return

        import pickle
        cache_pkl = path + ".fast_v8.pkl"
        try:
            st = os.stat(path)
            sig = (int(st.st_size), int(st.st_mtime))
            if os.path.exists(cache_pkl):
                with open(cache_pkl, "rb") as fc:
                    cached = pickle.load(fc)
                if cached.get("sig") == sig:
                    self.medians.update(cached["medians"])
                    self.stroke_counts.update(cached["stroke_counts"])
                    self.content_bboxes.update(cached["content_bboxes"])
                    GlyphRepository._GLOBAL_SKELETON_CACHE = (self.medians.copy(), self.stroke_counts.copy(), self.content_bboxes.copy())
                    logger.info(f"⚡ [SKELETON-CACHE HIT] Мгновенно загружено {len(cached['medians'])} векторных скелетов MakeMeAHanzi.")
                    return
        except Exception:
            sig = (0, 0)

        logger.info(f"Loading MakeMeAHanzi dataset: {path}")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    char = data.get("character")
                    raw_medians = data.get("medians", [])
                    if char and raw_medians:
                        compiled = [[(float(pt[0]), 900.0 - float(pt[1])) for pt in s] for s in raw_medians if s]
                        if compiled:
                            self._register_glyph(char, compiled)
                            count += 1
                except Exception:
                    continue
        logger.info(f"Indexed {count} character skeletons from MakeMeAHanzi.")
        GlyphRepository._GLOBAL_SKELETON_CACHE = (self.medians.copy(), self.stroke_counts.copy(), self.content_bboxes.copy())
        try:
            with open(cache_pkl, "wb") as fc:
                pickle.dump({
                    "sig": sig,
                    "medians": self.medians,
                    "stroke_counts": self.stroke_counts,
                    "content_bboxes": self.content_bboxes
                }, fc, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _register_glyph(self, char: str, strokes: List[List[Tuple[float, float]]]) -> None:
        self.medians[char] = strokes
        self.stroke_counts[char] = len(strokes)
        all_x = [pt[0] for s in strokes for pt in s]
        all_y = [pt[1] for s in strokes for pt in s]
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            self.content_bboxes[char] = BoundingBox(min_x, min_y, max(1.0, max_x - min_x), max(1.0, max_y - min_y))
        else:
            self.content_bboxes[char] = BoundingBox(0, 0, 1024, 1024)

    def has_glyph(self, char: str) -> bool:
        return char in self.medians

    def get_stroke_count(self, char: str) -> int:
        return self.stroke_counts.get(char, 4)

    def get_medians(self, char: str) -> Optional[List[List[Tuple[float, float]]]]:
        return self.medians.get(char)

    def get_content_bbox(self, char: str) -> BoundingBox:
        return self.content_bboxes.get(char, BoundingBox(0, 0, 1024, 1024))


class RadicalLeaf:
    def __init__(self, char: str, slot: SpatialSlot, repo: GlyphRepository):
        self.origin_char = char
        self.char = SafeAllographResolver.resolve(char, slot, set(repo.medians.keys()))
        self.medians = repo.get_medians(self.char)
        self.content_bbox = repo.get_content_bbox(self.char)
        self.repo = repo
        self.target_box = BoundingBox(0, 0, 1024, 1024)

    def get_stroke_count(self) -> int:
        return self.repo.get_stroke_count(self.char)

    def calculate_geometry(self, box: BoundingBox) -> None:
        pad = max(4.0, 16.0 - self.get_stroke_count() * 0.8)
        self.target_box = box.inset(pad, pad)

    def render(self, renderer: Any) -> None:
        renderer.draw_glyph_leaf(
            char=self.char,
            medians=self.medians,
            content_box=self.content_bbox,
            target_box=self.target_box,
            stroke_count=self.get_stroke_count()
        )


class CompositionNode:
    def __init__(self, operator: str, children: List[RadicalLeaf]):
        self.operator = operator
        self.children = children

    def calculate_geometry(self, box: BoundingBox) -> None:
        counts = [max(1, c.get_stroke_count()) for c in self.children]
        total = sum(counts)

        if self.operator == "⿰":
            left_w = 0.28 + 0.08 * (counts[0] / total)
            left_w = min(0.36, max(0.24, left_w))
            gap = box.width * 0.04
            w1 = (box.width - gap) * left_w
            w2 = box.width - w1 - gap
            self.children[0].calculate_geometry(BoundingBox(box.x, box.y, w1, box.height))
            self.children[1].calculate_geometry(BoundingBox(box.x + w1 + gap, box.y, w2, box.height))
        else:
            top_h = 0.28 + 0.08 * (counts[0] / total)
            top_h = min(0.36, max(0.24, top_h))
            gap = box.height * 0.04
            h1 = (box.height - gap) * top_h
            h2 = box.height - h1 - gap
            self.children[0].calculate_geometry(BoundingBox(box.x, box.y, box.width, h1))
            self.children[1].calculate_geometry(BoundingBox(box.x, box.y + h1 + gap, box.width, h2))

    def render(self, renderer: Any) -> None:
        for child in self.children:
            child.render(renderer)


class CalligraphicPillowRenderer:
    """Субпиксельный каллиграфический рендерер с нажимом кисти и автопоиском системных CJK-шрифтов."""
    _CACHED_FONT_PATH: Optional[str] = None
    _FONT_SEARCHED: bool = False

    def __init__(self, target_resolution: int = 512, supersample_factor: int = 2):
        self.target_res = target_resolution
        self.scale = supersample_factor
        self.canvas_res = target_resolution * supersample_factor
        self.image = Image.new("RGBA", (self.canvas_res, self.canvas_res), (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.ink_color = (18, 18, 18, 255)
        self._font_path = self._resolve_system_cjk_font()

    @classmethod
    def _resolve_system_cjk_font(cls) -> Optional[str]:
        if cls._FONT_SEARCHED:
            return cls._CACHED_FONT_PATH
        cls._FONT_SEARCHED = True

        candidate_paths = [
            os.getenv("CANGJIE_FONT_PATH", ""),
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc"
        ]
        for p in candidate_paths:
            if p and os.path.exists(p):
                cls._CACHED_FONT_PATH = p
                return p

        try:
            out = subprocess.check_output(["fc-list", ":lang=zh", "file"], text=True, timeout=3)
            for line in out.splitlines():
                path = line.split(":")[0].strip()
                if path and os.path.exists(path):
                    cls._CACHED_FONT_PATH = path
                    return path
        except Exception:
            pass

        logger.warning("Системный CJK-шрифт не найден! Установите: sudo apt install fonts-noto-cjk")
        return None

    def draw_glyph_leaf(
        self,
        char: str,
        medians: Optional[List[List[Tuple[float, float]]]],
        content_box: BoundingBox,
        target_box: BoundingBox,
        stroke_count: int
    ) -> None:
        tx = target_box.x * self.scale
        ty = target_box.y * self.scale
        tw = target_box.width * self.scale
        th = target_box.height * self.scale

        if medians and len(medians) > 0:
            cx, cy = content_box.x, content_box.y
            cw, ch = max(1.0, content_box.width), max(1.0, content_box.height)
            sx, sy = tw / cw, th / ch
            base_w = max(16.0, 36.0 - stroke_count * 0.9) * self.scale

            for stroke in medians:
                if len(stroke) < 2:
                    continue
                dense = []
                for i in range(len(stroke) - 1):
                    p0, p1 = stroke[i], stroke[i + 1]
                    x0, y0 = tx + (p0[0] - cx) * sx, ty + (p0[1] - cy) * sy
                    x1, y1 = tx + (p1[0] - cx) * sx, ty + (p1[1] - cy) * sy
                    seg_len = math.hypot(x1 - x0, y1 - y0)
                    sub_steps = max(6, int(seg_len * 2))
                    for step in range(sub_steps):
                        t = step / sub_steps
                        dense.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
                dense.append((tx + (stroke[-1][0] - cx) * sx, ty + (stroke[-1][1] - cy) * sy))

                n_pts = len(dense)
                for idx, (px, py) in enumerate(dense):
                    prog = idx / max(1, n_pts - 1)
                    if prog < 0.18:
                        pressure = 0.55 + 0.45 * (prog / 0.18)
                    elif prog > 0.82:
                        pressure = 1.0 - 0.55 * ((prog - 0.82) / 0.18)
                    else:
                        pressure = 1.0
                    r = (base_w * pressure) / 2.0
                    self.draw.ellipse([px - r, py - r, px + r, py + r], fill=self.ink_color)
        elif self._font_path:
            font_size = int(min(tw, th) * 0.90)
            font = ImageFont.truetype(self._font_path, font_size)
            bbox = self.draw.textbbox((0, 0), char, font=font)
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ox = tx + (tw - bw) / 2.0 - bbox[0]
            oy = ty + (th - bh) / 2.0 - bbox[1]
            self.draw.text((ox, oy), char, font=font, fill=self.ink_color)
        else:
            # Геометрический контурный fallback, исключающий пустой белый холст при отсутствии шрифтов
            w_line = max(4, int(min(tw, th) * 0.06))
            self.draw.rectangle([tx + tw * 0.15, ty + th * 0.15, tx + tw * 0.85, ty + th * 0.85], outline=self.ink_color, width=w_line)
            self.draw.line([tx + tw * 0.5, ty + th * 0.1, tx + tw * 0.5, ty + th * 0.9], fill=self.ink_color, width=w_line)

    def finalize(self) -> Image.Image:
        return self.image.resize((self.target_res, self.target_res), Image.Resampling.LANCZOS)


class SVGVectorRenderer:
    def __init__(self, resolution: int = 512):
        self.res = resolution
        self.paths: List[str] = []

    def draw_glyph_leaf(
        self,
        char: str,
        medians: Optional[List[List[Tuple[float, float]]]],
        content_box: BoundingBox,
        target_box: BoundingBox,
        stroke_count: int
    ) -> None:
        tx, ty, tw, th = target_box.x, target_box.y, target_box.width, target_box.height

        if medians and len(medians) > 0:
            cx, cy, cw, ch = content_box.x, content_box.y, content_box.width, content_box.height
            sx, sy = tw / max(1.0, cw), th / max(1.0, ch)
            width = max(10.0, 24.0 - stroke_count * 0.6)

            for stroke in medians:
                if not stroke:
                    continue
                d = [f"M {tx + (stroke[0][0] - cx) * sx:.2f} {ty + (stroke[0][1] - cy) * sy:.2f}"]
                for pt in stroke[1:]:
                    d.append(f"L {tx + (pt[0] - cx) * sx:.2f} {ty + (pt[1] - cy) * sy:.2f}")
                self.paths.append(
                    f'<path d="{" ".join(d)}" fill="none" stroke="#121212" '
                    f'stroke-width="{width:.2f}" stroke-linecap="round" stroke-linejoin="round" />'
                )
        else:
            font_size = int(min(tw, th) * 0.85)
            self.paths.append(
                f'<text x="{tx + tw/2.0:.2f}" y="{ty + th*0.82:.2f}" font-size="{font_size}" '
                f'text-anchor="middle" font-family="sans-serif, Noto Sans CJK" fill="#121212">{char}</text>'
            )

    def to_svg(self) -> str:
        body = "\n  ".join(self.paths)
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.res} {self.res}" '
            f'width="{self.res}" height="{self.res}">\n'
            f'  <rect width="100%" height="100%" fill="#ffffff" />\n'
            f'  {body}\n'
            f'</svg>'
        )


EXISTING_COMBINATIONS: Set[Tuple[str, str, str]] = {
    ("⿰", "氵", "木"), ("⿰", "亻", "木"), ("⿰", "氵", "青"),
    ("⿰", "日", "月"), ("⿰", "木", "木"), ("⿰", "火", "尧"),
    ("⿰", "氵", "尧"), ("⿰", "氵", "仓"), ("⿰", "钅", "仓"),
    ("⿰", "木", "兆"), ("⿰", "扌", "兆"), ("⿰", "扌", "甫"),
    ("⿰", "木", "甫"), ("⿰", "日", "青"), ("⿰", "忄", "青"),
    ("⿰", "讠", "青"), ("⿰", "亻", "寺"), ("⿰", "扌", "寺"),
    ("⿰", "讠", "寺"), ("⿰", "彳", "寺")
}