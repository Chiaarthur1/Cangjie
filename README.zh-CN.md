# 仓颉项目 (Project Cangjie)

> **AI 古文字识别系统**：上传古代汉字（甲骨文/金文/篆书/楷书）照片，系统回答
> **这是什么字、古代怎么读、是什么意思** —— 依据汉字构形规律（六书）推导，
> 而非神经网络「凭空想象」。

系统识别**甲骨文、金文、篆书、楷书**（甲骨、青铜器、拓片、纸张），并输出一份
可解释的「古文字档案」，所有结论均锚定于真实数据源（Unihan、CHISE IDS、
Baxter–Sagart、知识图谱）。

---

## ✨ 核心特性

- **照片 → 档案**：上传古文字照片（可附出土语境：年代、载体、地点），得到字形、
  读音（拼音 + 中古音/上古音）、字义、结构拆解（IDS）与字形结构。
- **防幻觉**：档案中的每条结论都对应知识图谱中的一条边（`构成`、`读音`、`释义`、
  `谐声`、`先例`）并注明来源；LLM 严格依据该列表推理，而非臆测。
- **诚实**：无法识别时返回 `UNRECOGNIZED`，而不是随机猜测；指标实测、非宣称。
- **模型可选**：检测器（`auto|rtdetr|yolo|pixel|none`）、匹配器
  （`chamfer_hog|embedding`）、LLM（`none|ollama|vllm`）。无 GPU/权重时诚实回退到
  像素级路径。
- **完整控制面板**（7 个标签页）：解码器、工作室、智能体辩论、模型、训练、数据、
  评测+设置。
- **无需 GPU 即可运行**：像素级路径（Chamfer + HOG）与 CPU 向量在任意电脑上可用；
  神经网络路径在有训练权重时自动启用。

---

## 🚀 安装

```bash
cd cangjie_project

# 1) 虚拟环境
python3 -m venv .venv && source .venv/bin/activate

# 2) 基础依赖（覆盖 95% 功能）
pip install numpy pillow opencv-python networkx

# 3) 训练检测器/向量时
pip install torch torchvision ultralytics
```

数据（Unihan、CHISE IDS、Baxter–Sagart、MakeMeAHanzi）位于 `data/`，也可用
`download_datasets.sh` 下载。

---

## 🎬 快速开始

### 识别古文字照片

```bash
python3 cangjie_photo_decoder.py \
    --photo photo.jpg \
    --context "占卜甲骨，商代，安阳" \
    --output ./out --json
```

输出 JSON 档案：`recognition_status`、`reconstructed_deep_ids`、
`paleographic_decoding`（读音/释义/依据）、`knowledge_graph_context`、
`model_selection`。

### 启动 Web 面板

```bash
python3 cangjie_ui_studio.py --serve --port 8080
# → http://localhost:8080
```

### 统一配置

```bash
python3 cangjie_config.py --show            # 查看全部配置
python3 cangjie_config.py --machine auto    # 自动选择硬件
python3 cangjie_config.py --set detector.backend=pixel
python3 cangjie_config.py --validate
```

---

## 🎛️ 控制面板

| 标签页 | 用途 |
|---|---|
| **1. 解码器** | 照片/IDS → 档案 |
| **2. 工作室** | 照片扫描、手绘笔画与部件、模型选择 |
| **3. 智能体辩论** | 假设生成器 vs 校验者（依据 Unihan/CHISE 核验） |
| **4. 模型** | 模型注册表 + 检测器/匹配器/LLM 热切换 |
| **5. 训练** | 后台任务（损失 sanity / 向量 / 检测器）+ 实时日志 |
| **6. 数据** | 数据集注册表 + 标注（自动检测 → YOLO） |
| **7. 评测+设置** | 诚实评测、重建模板库、查看配置 |

HTTP API：`/api/decode_photo`、`/api/system_status`、`/api/agent_duel`、`/api/config`、
`/api/models`、`/api/models/switch`、`/api/train[/status|/list]`、`/api/datasets`、
`/api/eval`、`/api/rebuild`、`/api/label/export`。

---

## 🧠 工作原理（简述）

```
照片 → 预处理（Sauvola 二值化、拓片反相）
     → 多字切分（卷轴/成行）
     → 部件检测（RT-DETR / 像素递归分割）
     → 部件识别（Chamfer+HOG 带校准 / 向量检索）
     → IDS 结构组装（⿰⿱⿲…）+ 整字直接识别
     → 六书推导（字义 + 中古/上古读音）
     → 知识图谱 grounded 语境
```

---

## 📁 项目结构

| 模块 | 用途 |
|---|---|
| `cangjie_photo_decoder.py` | 主流水线「照片 → 档案」 |
| `cangjie_ui_studio.py` | Web 面板 + HTTP API |
| `cangjie_config.py` | 统一配置（硬件档位 + 模型选择） |
| `cangjie_model_registry.py` | 模型注册表 + 热切换 |
| `cangjie_train_manager.py` | 后台训练任务 |
| `cangjie_deducer.py` | 六书推导 |
| `cangjie_knowledge_graph.py` | 知识图谱 + grounded 语境 |
| `cangjie_paleo_matcher.py` + `cangjie_calibration.py` | 部件匹配 + 置信度校准 |
| `cangjie_layout_to_tree.py` | 几何 → IDS 树 |
| `cangjie_metric_embed.py` | 度量学习（向量 + kNN） |
| `cangjie_rtdetr_train.py` | 检测器训练 |
| `cangjie_dataset_manager.py` / `cangjie_labeler.py` | 数据集注册表 / 标注 |
| `cangjie_obimd_loader.py` | OBIMD → YOLO 转换 |
| `cangjie_honest_eval.py` / `cangjie_tree_eval.py` | 诚实评测 |
| `cangjie_rebuild_banks.py` | 重建模板库 |

---

## 📊 实测指标

| 指标 | 数值 |
|---|---|
| IDS 解析 round-trip（CHISE） | **1.0000**（163 059 条复合记录） |
| 诚实评测 top-1 | **0.978**（合成退化样本） |
| 误判率（噪声→字形） | **1.7%**（3/180） |
| 置信度校准 | acc 0.9912 / prec 0.9991 |

> ⚠️ 上述数字是合成数据上的「上限」。真实馆藏照片的诚实指标需在 OBIMD 上训练
> 检测器后得出。

---

## ⚖️ 数据许可

本项目使用多来源数据，各遵循其许可：

- **Unihan**（Unicode）— Unicode License；
- **CHISE IDS** — CHISE 自由许可；
- **Baxter–Sagart** — CC BY 4.0；
- **Make Me a Hanzi** — Arphic Public License；
- **OBIMD** — CC BY 4.0（需注明出处）。

> 商用前请对依赖与数据进行许可审计（项目内置数据集许可校验器）。

---

## 📌 状态

✅ 可用：主流水线、Web 面板、模型选择、知识图谱、质量评测。

⏳ 需要 GPU + 数据：检测器与向量训练（代码已就绪）、整字模板库（渲染约 1.5 小时）、
OBIMD 与馆藏照片（尚未下载）。

---

为古文字学家、语言学家与古代汉字研究者而作 ❤️
