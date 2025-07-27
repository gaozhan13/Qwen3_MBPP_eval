# Qwen3 MBPP 评估框架

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.20+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Qwen3 模型代码生成能力评估**

本项目提供了一个MBPP（Mostly Basic Python Problems）评估框架，专门用于评估Qwen3 模型（包括Base和Instruct版本）在代码生成任务上的性能表现。

## 🎯 项目特性

- **双模型支持**：同时支持Qwen3 Base模型和Instruct模型评估
- **Pass@K评估**：实现完整的pass@k评估指标
- **零样本评估**：采用0-shot评估方式，真实反映模型的代码生成能力
- **安全执行**：集成超时机制，防止生成代码陷入死循环
- **详细报告**：生成完整的评估报告和详细的测试结果

## 📊 评估指标

- **Pass@K**：测量模型生成的K个候选解决方案中至少有一个正确的概率
- **平均执行时间**：单个样本的平均生成和测试时间
- **代码长度统计**：生成代码的长度分布统计
- **成功率分析**：按任务类型和难度级别的成功率分析

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 1.9+
- CUDA（可选，用于GPU加速）
- 至少8GB内存（推荐16GB）

### 安装依赖

1. **克隆项目**：
```bash
git clone <your-repository-url>
cd qwen3_eval
```

2. **自动安装**（推荐）：
```bash
bash install_evaluation_frameworks.sh
```

3. **手动安装**：
```bash
pip install -r requirements.txt
```

### 基础使用

#### 评估Base模型（预训练模型）

```bash
# Pass@1评估（最常用）
python run_mbpp_original_base.py --model Qwen/Qwen3-0.6B-Base --k 1

# Pass@10评估
python run_mbpp_original_base.py --model Qwen/Qwen3-0.6B-Base --k 10

# 快速测试（仅评估前50个样本）
python run_mbpp_original_base.py --model Qwen/Qwen3-0.6B-Base --k 5 --max-samples 50
```

#### 评估Instruct模型（后训练模型）

```bash
# Pass@1评估
python run_mbpp_original_instruct_non_thinking.py --model Qwen/Qwen3-0.6B --k 1

# Pass@10评估
python run_mbpp_original_instruct_non_thinking.py --model Qwen/Qwen3-0.6B --k 10

# 自定义输出目录
python run_mbpp_original_instruct_non_thinking.py --model Qwen/Qwen3-0.6B --k 1 --output-dir custom_results
```

## ⚙️ 配置参数

### 命令行参数

| 参数 | Base模型默认值 | Instruct模型默认值 | 说明 |
|------|----------------|-------------------|------|
| `--model` | `Qwen/Qwen3-0.6B-Base` | `Qwen/Qwen3-0.6B` | 模型路径或HuggingFace模型名 |
| `--k` | `1` | `1` | Pass@K评估中的K值 |
| `--output-dir` | `mbpp_results_base` | `mbpp_results_instruct_non_thinking` | 结果输出目录 |
| `--max-samples` | `None` | `None` | 最大评估样本数（用于测试） |
| `--max-length` | `512` | `512` | 最大生成token数量 |
| `--timeout` | `5` | `5` | 代码执行超时时间（秒） |
| `--log-level` | `INFO` | `INFO` | 日志级别 |

### 生成配置

#### Base模型配置
```python
temperature: 0.2
do_sample: True
top_p: 0.95
top_k: 50
repetition_penalty: 1.05
```

#### Instruct模型配置
```python
temperature: 0.7
top_p: 0.8
top_k: 20
min_p: 0.0
```

## 📁 项目结构

```
qwen3_eval/
├── README.md                                    # 项目说明文档
├── requirements.txt                             # Python依赖列表
├── install_evaluation_frameworks.sh             # 自动安装脚本
├── run_mbpp_original_base.py                    # Base模型评估脚本
├── run_mbpp_original_instruct_non_thinking.py   # Instruct模型评估脚本
├── mbpp_results_base/                           # Base模型评估结果
│   ├── mbpp_base_evaluation_results.json       # 评估摘要
│   └── mbpp_base_detailed_results.json         # 详细结果
└── mbpp_results_instruct_non_thinking/          # Instruct模型评估结果
    ├── mbpp_instruct_evaluation_results.json   # 评估摘要
    └── mbpp_instruct_detailed_results.json     # 详细结果
```

## 📈 结果文件说明

### 评估摘要（`*_evaluation_results.json`）

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "model_type": "instruct",
  "evaluation_method": "0-shot",
  "dataset": "mbpp_original",
  "k": 1,
  "total_samples": 500,
  "passed_samples": 250,
  "pass_at_k": 0.500,
  "evaluation_time": 1800.0,
  "average_time_per_sample": 3.6
}
```

### 详细结果（`*_detailed_results.json`）

包含每个样本的详细信息：
- 任务ID和提示
- 生成的代码（K次生成）
- 测试用例和结果
- 代码长度和生成时间
- 是否通过测试
