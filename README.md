<div align="center">

# 🀄 Project Cangjie (仓颉计划)
### Neuro-Symbolic AI Pipeline for Zero-Hallucination Chinese Character Decomposition & Paleographic Semantic Deduction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Vision](https://img.shields.io/badge/Vision-RT--DETR_%7C_YOLOv10-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Neo4j](https://img.shields.io/badge/Knowledge_Graph-Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com)
[![VLM](https://img.shields.io/badge/VLM-Qwen2--VL_%7C_DeepSeek--VL-7B2CBF?style=for-the-badge)](https://github.com/QwenLM/Qwen2-VL)

</div>

---

## 📌 Overview & Motivation
Modern Vision-Language Models (VLMs) frequently suffer from **semantic hallucinations** when encountering rare, historical (Oracle Bone / Bronze / Seal scripts), or out-of-distribution (OOD) Chinese characters (汉字). Instead of abstaining or performing rigorous structural analysis, standard models fabricate plausible-sounding meanings based on visual similarity to common characters.

**Project Cangjie** solves this by decoupling **structural perception** from **semantic reasoning** through a three-stage **Neuro-Symbolic Architecture**:
1. **Sub-character Visual Decomposition** into 214 Kangxi radicals (康熙部首) and 12 Ideographic Description Sequence (IDS) spatial operators.
2. **Deterministic Graph Grounding** in `Neo4j` using classical etymological dictionaries (*Shuowen Jiezi* 说文解字, CHISE, and Unihan).
3. **Constrained Deductive Synthesis** via Multimodal LLMs (`Qwen2-VL`), explicitly separating semantic radicals (`形旁`) from phonetic components (`声旁`).

---

## 🏗 System Architecture

```text
+-----------------------------------------------------------------------------------+
|                         INPUT: Rare / Ancient / Pseudo Glyph                      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|  1. VISION MODULE (YOLOv10 / RT-DETR)                                             |
|  • Detects 214 Kangxi Radicals + 12 Spatial IDS Operators (⿰, ⿱, ⿲, ⿳, ⿴, etc.) |
|  • Constructs Structural Bounding-Box Hierarchy & IDS Tree                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|  2. ETYMOLOGICAL KNOWLEDGE GRAPH (Neo4j)                                          |
|  • Queries CHISE IDS, Unihan & Shuowen Jiezi (说文解字)                             |
|  • Traverses explicit [:SEMANTIC] (形旁) vs. [:PHONETIC] (声旁) edges              |
|  • Exports deterministic JSON etymological context                                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|  3. SEMANTIC SYNTHESIZER (Qwen2-VL / DeepSeek-VL)                                 |
|  • Performs deductive semantic hypothesis generation                              |
|  • Strict Anti-Hallucination Gate: returns NULL on ungrounded pseudo-characters   |
+-----------------------------------------------------------------------------------+
