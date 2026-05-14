# LAAT - 基于大语言模型的算法编程学习辅助系统

**LLM Assistant Algorithm Teaching**

基于 LangGraph 的算法竞赛对拍验证系统，通过大语言模型自动生成暴力代码、数据生成器、优化解法，并对拍验证正确性。支持本地 Web 部署，可求解自定义题目、APPS 数据集题目、Codeforces 题目，并支持大模型翻译英文题面。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| **对拍验证求解** | 暴力代码 → 数据生成器 → 优化解法，三代码自动对拍，发现逻辑错误自动重试 |
| **自定义题目提交** | 在 Web 界面输入题面、输入输出格式、样例，一键求解 |
| **APPS 数据集** | 直接加载 APPS train/test 数据集，批量或单题求解，自动跳过已求解题目 |
| **Codeforces 题库** | 从 Codeforces API 浏览题目，按 Rating/Tag 筛选，一键导入 + 翻译 + 求解 |
| **大模型翻译** | 将 Codeforces 英文题面翻译为中文，保留数学公式和变量名 |
| **实时进度** | WebSocket 推送求解日志，前端实时显示每个阶段（生成/验证/重试）的进展 |
| **持久化存储** | 每道题的代码、结果、日志均保存到本地文件夹，重启不丢失 |
| **运行时配置** | Web 界面直接修改 LLM 模型、API Key、Base URL 等参数，无需重启 |

---

## 系统架构

```
用户题面 / APPS / Codeforces
         │
         ▼
┌─────────────────────────────────────────────┐
│           LangGraph 对拍求解图                │
│                                             │
│  START → load_problem                       │
│    → generate_bf → validate_bf ──(重试)──┐  │
│    → generate_dm → validate_dm ──(重试)──┤  │
│    → generate_tests                        │  │
│    → generate_sol → validate_sol ─(重试)─┘  │
│    → END                                   │
│                                             │
│  每个阶段失败自动重试（最多 3 次）            │
│  验证方式：样例测试 + 对拍测试               │
└─────────────────────────────────────────────┘
         │
         ▼
  保存到题目文件夹:
  ├── brute_force.py    # 暴力代码
  ├── data.py           # 数据生成器
  ├── solution.py       # 优化解法
  ├── result.json       # 求解结果
  └── execution_log.txt # 执行日志
```

---

## 快速开始

### 环境要求

- Python 3.10+
- 操作系统：Windows / macOS / Linux
- 网络：需要访问 LLM API（默认 DeepSeek）

### 1. 克隆项目

```bash
git clone <repo-url>
cd LLM-AssistantAlgorithmTeaching
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：

| 包名 | 用途 |
|------|------|
| `langgraph` | 求解图状态机 |
| `langchain-openai` | LLM 调用（兼容 OpenAI API） |
| `langchain-core` | LangChain 基础组件 |
| `openai` | OpenAI API 客户端 |
| `tqdm` | 批量求解进度条 |
| `fastapi` | Web 后端 |
| `uvicorn[standard]` | ASGI 服务器 |
| `websockets` | WebSocket 实时推送 |

### 3. 配置 LLM API

有两种方式配置：

**方式 A：环境变量（推荐，最安全）**

```bash
# Linux / macOS
export LLM_API_KEY="your-api-key-here"
export LLM_MODEL="deepseek-chat"           # 可选，默认 deepseek-v4-flash
export LLM_BASE_URL="https://api.deepseek.com/v1"  # 可选

# Windows PowerShell
$env:LLM_API_KEY = "your-api-key-here"
$env:LLM_MODEL = "deepseek-chat"
$env:LLM_BASE_URL = "https://api.deepseek.com/v1"
```

**方式 B：Web 界面设置**

启动服务器后，点击页面右上角 ⚙️ 设置按钮，在弹窗中填入 API Key、模型名、Base URL，点击保存即可。修改立即生效，无需重启。

> 支持所有兼容 OpenAI API 的模型服务（DeepSeek、OpenAI、智谱、月之暗面等），只需修改 `base_url` 和 `model`。

### 4. 启动服务

```bash
python server.py
```

启动成功后会显示：

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

打开浏览器访问 **http://localhost:8000** 即可使用。

---

## 使用指南

### 提交问题

1. 点击「提交问题」标签
2. 填写题面描述、输入输出格式、数据范围
3. 添加至少一组样例（输入 + 输出）
4. 设置最大重试次数和生成测试数据数量
5. 点击「开始求解」
6. 等待求解完成，可在日志面板实时查看进度
7. 求解完成后点击「查看」可切换查看三种代码（暴力 / 数据生成器 / 优化解法）

### APPS 数据集

1. 点击「APPS数据集」标签
2. 选择 Train / Test 分割
3. 搜索或浏览题目列表
4. 点击「求解」对单题求解，或点击「刷新」重新加载列表
5. 已求解的题目会显示结果（PASS / FAIL），可点击「查看」或「重跑」

> APPS 数据集需自行下载并放置在 `APPS/train/` 和 `APPS/test/` 目录下。首次使用时会自动转换格式。

### Codeforces 题库

1. 点击「Codeforces」标签
2. 使用筛选器选择题目难度范围（Rating）、标签（Tag）
3. 点击「搜索」从 Codeforces API 加载题目列表
4. 对每道题目有三个操作按钮：
   - **导入**：抓取题目描述和样例，保存到本地
   - **翻译**：导入后调用 LLM 将英文题面翻译为中文，翻译结果显示在页面下方
   - **求解**：导入后直接运行对拍求解流程，日志实时显示

> 部分题目页面可能被 Cloudflare 拦截导致导入失败，系统会自动尝试正则提取作为回退方案。

### 求解历史

1. 点击「求解历史」标签
2. 查看所有已求解题目的列表（来源、结果、时间）
3. 点击「查看」查看代码详情，点击「重跑」强制重新求解

### 设置

点击右上角 ⚙️ 齿轮按钮，可修改：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 模型名称 | LLM 模型 ID | `deepseek-v4-flash` |
| API Base URL | API 服务地址 | `https://api.deepseek.com/v1` |
| API Key | 认证密钥（仅显示末 4 位） | 环境变量或默认值 |
| Max Tokens | 最大生成 token 数 | `4096` |
| Temperature | 生成温度 | `0.7` |

---

## 项目结构

```
LLM-AssistantAlgorithmTeaching/
├── server.py                 # FastAPI 后端（API + WebSocket + 静态文件）
├── requirements.txt          # Python 依赖
├── web/
│   └── index.html            # 前端单文件应用（Vue 3 + Tailwind CSS）
│
├── src/
│   ├── config.py             # 全局配置（LLM 参数、超时、路径）
│   ├── runner.py             # 批量运行器
│   │
│   ├── graph/
│   │   ├── builder.py        # LangGraph 图构建 + invoke_solve()
│   │   ├── state.py          # ProblemState TypedDict（23 个字段）
│   │   ├── routing.py        # 条件边路由（重试/失败/通过）
│   │   └── nodes/
│   │       ├── load_problem.py     # 加载题目
│   │       ├── brute_force.py      # 生成 + 验证暴力代码
│   │       ├── data_maker.py       # 生成 + 验证数据生成器
│   │       ├── test_gen.py         # 用数据生成器生成测试数据
│   │       └── solution.py         # 生成 + 验证优化解法
│   │
│   ├── llm/
│   │   ├── codegen.py        # LLM 代码生成（暴力/数据/解法）
│   │   ├── compare.py        # 输出对比（token 级别）
│   │   └── translate.py      # LLM 题面翻译（EN → CN）
│   │
│   └── converters/
│       ├── apps_converter.py  # APPS 数据集格式转换
│       └── cf_scraper.py      # Codeforces 题目爬取（API + HTML）
│
├── APPS/                     # APPS 数据集（需自行下载）
│   ├── train/
│   └── test/
├── APPS_converted/           # 转换后的 APPS 题目
├── cf_problems/              # 导入的 Codeforces 题目
├── user_problems/            # 用户自定义题目
├── results/                  # 批量运行结果
└── .cf_cache/                # Codeforces API 缓存
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/problems` | 列出所有已求解题目 |
| `GET` | `/api/problems/{id}` | 获取题目详情 + 三种代码 |
| `POST` | `/api/solve` | 提交自定义题目求解 |
| `POST` | `/api/solve-apps/{id}` | 求解 APPS 题目 |
| `GET` | `/api/apps-list` | 获取 APPS 题目列表 |
| `GET` | `/api/task/{id}` | 查询求解任务状态 |
| `GET` | `/api/config` | 获取当前 LLM 配置（API Key 脱敏） |
| `PUT` | `/api/config` | 更新 LLM 配置 |
| `GET` | `/api/cf/problems` | 获取 Codeforces 题目列表（支持筛选） |
| `POST` | `/api/cf/import` | 导入 Codeforces 题目 |
| `POST` | `/api/cf/translate` | 翻译题面（EN → CN） |
| `POST` | `/api/cf/solve` | 导入并求解 Codeforces 题目 |
| `WS` | `/ws/solve/{task_id}` | WebSocket 实时推送求解日志 |

---

## 求解流程详解

系统采用竞赛中常用的**对拍验证**方法：

```
题面描述
   │
   ├──→ 1. 生成暴力代码（brute_force.py）
   │       时间复杂度高但正确性可信
   │       用样例验证，失败则重试（最多 3 次）
   │
   ├──→ 2. 生成数据生成器（data.py）
   │       生成随机测试数据
   │       用样例验证输出格式
   │
   ├──→ 3. 生成测试数据
   │       运行数据生成器，产生 20 组测试输入
   │
   └──→ 4. 生成优化解法（solution.py）
           时间复杂度优化的代码
           先用样例验证，再用暴力代码对拍
           对拍失败则重试（最多 3 次）
```

**对拍逻辑**：对每组测试输入，分别运行暴力代码和优化解法，对比输出。若所有输出一致，则判定 PASS；否则判定 FAIL 并自动重试。

---

## 环境变量参考

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_MODEL` | 模型名称 | `deepseek-v4-flash` |
| `LLM_BASE_URL` | API Base URL | `https://api.deepseek.com/v1` |
| `LLM_API_KEY` | API 密钥 | - |
| `LLM_MAX_TOKENS` | 最大生成 token | `4096` |
| `LLM_TEMPERATURE` | 生成温度 | `0.7` |

---

## 常见问题

### Q: 启动后提示 `ModuleNotFoundError`

确保在项目根目录运行 `pip install -r requirements.txt`，且 Python 版本 ≥ 3.10。

### Q: 求解一直失败

- 检查 API Key 是否正确（设置按钮 → 重新输入）
- 检查网络能否访问 API 地址
- 尝试降低题目难度或增加重试次数
- 查看日志面板中的错误信息

### Q: Codeforces 题目导入失败

部分 CF 题目页面有 Cloudflare 验证，可能导致抓取失败。可以：
- 重试导入
- 使用较老/较简单的题目（如 4A、71A 等）
- 手动将题目信息写入 `cf_problems/` 目录

### Q: Windows 下中文乱码

本系统使用 UTF-8 编码。如果终端显示乱码，是终端编码问题，不影响 Web 界面和文件存储。Web 界面中所有中文显示正常。

### Q: 如何更换 LLM 服务商

点击设置按钮，修改：
- **模型名称**：如 `gpt-4o`、`glm-4-flash`、`moonshot-v1-8k`
- **API Base URL**：对应服务商的 API 地址
- **API Key**：对应的密钥

只要服务商兼容 OpenAI API 格式即可使用。

---

## 技术栈

- **后端**：Python 3.10+ / FastAPI / LangGraph / LangChain
- **前端**：Vue 3 (CDN) / Tailwind CSS / highlight.js
- **LLM**：兼容 OpenAI API 的任意模型服务
- **实时通信**：WebSocket（主）+ HTTP 轮询（降级）

---

