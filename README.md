# StardewPal · 星露谷物语多模态问答助手

一个**能用截图提问**的游戏知识库问答系统。基于社区 Wiki 构建，覆盖 OCR、图像匹配、
混合检索与工具计算，并提供自动生成的评测集与可观测性。

> 定位：面向「AI 会说但说不准」的场景 —— 精确数值、品质加成、加工倍率、跨表计算。

---

## 为什么做这个

直接用大模型问「金星蓝莓能卖多少钱」，它会给一个看起来合理的数字，但经常是错的，
而且**错的时候语气一样自信**。原因有三：

1. **数值记忆不可靠** —— 模型不知道 35 金、×1.5、×3.0 这些具体机制常量从哪来
2. **不会真的算** —— 多步倍率相乘时直接给个"像样"的结果
3. **无法核实** —— 说完没有任何依据可查

这个项目把三件事拆开做，各用最合适的工具：

| 环节 | 谁来做 | 为什么 |
|---|---|---|
| 找资料 | 混合检索（BM25 + 向量 + RRF） | 语义匹配，可引用 |
| 算数字 | **确定性工具函数** | 算术不能交给语言模型 |
| 说话 | 大模型 | 只负责组织语言和解释 |

---

## 架构

```
截图 / 文字
    │
    ├── 视觉链路 ────────────────────────────────┐
    │   ① OCR（RapidOCR，本地 CPU）               │  文字信号为主
    │   ② 图标匹配（pHash 召回 + 多特征精排）      │  视觉信号为辅
    │   ③ VLM（界面判断 + 低置信选择题兜底）       │
    │   ④ 置信度路由 → 高置信直答 / 中置信问模型 / 低置信让用户确认
    │                                            │
    └── 文本链路 ────────────────────────────────┤
        ① 路由：这问题要不要检索？                 │
        ② 混合检索：BM25 + 向量 → RRF 融合         │
        ③ 自省：资料够不够？不够就改写重查（多轮）   │
        ④ 计算：数值题强制走工具                    │
        ⑤ 生成：只用资料作答 + 引用编号             │
        ⑥ 门控：资料不相关 → 拒绝回答               │
                                                 ▼
                                          带引用的回答
```

### 关键设计决策（都有实测依据）

**1. 为什么 OCR 优先于图像匹配**

在 320 个真实图标上做的对照实验：

| 输入条件 | 像素特征 Top-1 | pHash Top-1 |
|---|---|---|
| 干净图标 | 100.0% | 94.7% |
| 真实扰动（8x缩放+模糊+JPEG+亮度） | 35.0% | 19.7% |

小尺寸像素图信息量不足，纯图像匹配不可靠。而 OCR 对中文 UI 文字几乎 100% 正确。
所以：**文字为主，图像为辅**，图像只提供候选，不做最终裁决。

**2. 为什么算术必须是工具**

「金星果酒」= 基础卖价 × 3.0（果酒）× 1.5（品质）。
三个数字错一个就全错。让模型心算等于放弃可控性。

**3. 为什么需要相关性门控**

资料存在 ≠ 资料相关。问「原神胡桃的倍率」，检索总会返回一些片段，
如果不做相关性判断，模型会拿星露谷的资料硬答。宁可拒答。

---

## 实测数据

索引规模（BWiki 全站抓取）：

```
页面          5,259
有效文档      2,002
文本块        5,948
物品实体      521
图标          1,200
```

嵌入模型：`bge-small-zh-v1.5` ONNX INT8 量化版，**23.3 MB**，纯 CPU，
实测 1024 条/秒（短文本），512 维。

评测集自动生成 164 条，覆盖 6 种题型：
单事实 / 品质计算 / 加工计算 / 属性 / 跨文档对比 / 拒答。

> 具体指标见 `backend/data/runs/eval-*.json`，由 `run_eval` 产出。

---

## 快速开始

```bash
# 1. 依赖
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# 2. 配置（不填也能跑，会自动进入 mock 模式）
copy .env.example .env

# 3. 数据管道
python -m backend.pipeline.fetch_model     # 下载嵌入模型（23MB）
python -m backend.pipeline.crawl           # 抓取 wiki（约 1.5 小时，限速 1 req/s）
python -m backend.pipeline.build_index     # 构建索引
python -m backend.pipeline.build_icons     # 构建图标索引

# 4. 启动
python -m backend.app.serve
# 打开 http://127.0.0.1:7860
```

### 评测

```bash
python -m backend.pipeline.run_eval --limit 60
python -m backend.pipeline.run_eval --no-agent    # 消融：关闭多轮决策
```

---

## 目录结构

```
backend/
  app/
    agent.py          决策层：路由 / 自省 / 工具 / 门控
    retrieval.py      混合检索 + RRF + 标题亲和度
    store.py          SQLite + FTS5 + sqlite-vec
    embedding.py      ONNX 本地嵌入
    llm.py            LLM 客户端（含 mock）
    vision/
      ocr.py          OCR + 版面还原 + 粘连切分
      icons.py        图标特征索引与匹配
      pipeline.py     视觉链路编排与置信度路由
    tools/calc.py     确定性计算工具
  pipeline/
    crawl.py          wiki 抓取与清洗
    chunker.py        切分（含数值注入）
    build_index.py    索引构建
    eval_set.py       评测集生成与评分
    run_eval.py       评测运行器
frontend/
  index.html          单页界面
  static/             样式与脚本（原生 JS，无构建步骤）
```

---

## 已知限制

- **图标识别上限**：48×48 像素图标信息量有限，且部分图标被多个物品复用
  （实测存在像素完全相同的图标对），所以只输出 Top-K 候选
- **语料规模**：BWiki 中文站约 2,500 条目，覆盖面小于英文官方 wiki
  （官方站已封锁 API，本项目不绕过）
- **演示模式**：未配置 API Key 时回答由检索片段拼接，用于验证链路而非展示效果

---

## 版权

语料来自 [BWiki 星露谷物语](https://wiki.biligame.com/stardewvalley/)，
内容遵循 CC BY-NC-SA 协议。本项目为学习与技术展示用途，**不做商业使用**。
抓取严格限速 1 请求/秒。
