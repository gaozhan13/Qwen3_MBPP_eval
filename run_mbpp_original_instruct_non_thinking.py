#!/usr/bin/env python3
"""
MBPP原版评估脚本 - Instruct模型版本
专门针对Qwen3后训练模型的评估。
该模块提供了完整的MBPP数据集评估框架，支持0-shot代码生成pass@k评估。

Example:
    基本使用方法（pass@1）：
    ```bash
    python run_mbpp_original_instruct_non_thinking.py --model Qwen/Qwen3-0.6B --k 1
    ```

    pass@10评估：
    ```bash
    python run_mbpp_original_instruct_non_thinking.py --model Qwen/Qwen3-0.6B --k 10
    ```

    限制样本数量进行快速测试：
    ```bash
    python run_mbpp_original_instruct_non_thinking.py --model Qwen/Qwen3-0.6B --max-samples 10 --k 5
    ```
"""

from __future__ import annotations

import re
import argparse
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import datasets
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ================================ 配置和常量 ================================


@dataclass(frozen=True)
class GenerationConfig:
    """代码生成配置类。

    包含用于控制模型生成行为的参数。
    """

    # Best practices from huggingface
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0


@dataclass
class EvaluationConfig:
    """评估配置类。
    包含评估过程中的所有可配置参数。
    """

    model_path: str
    k: int = 1  # pass@k中的k值
    output_dir: str = "mbpp_results_instruct_non_thinking"
    max_samples: int | None = None
    max_new_tokens: int = 512
    timeout_seconds: int = 5
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)


class EvaluationResult(NamedTuple):
    """单个评估样本的结果。"""

    task_id: int
    prompt: str
    generated_codes: list[str]  # 存储k次生成的代码
    test_cases: list[str]
    test_results: list[bool]  # 存储k次测试的结果
    test_passed: bool  # 是否至少有一次通过（pass@k）
    code_lengths: list[int]  # 存储k次生成代码的长度
    generation_times: list[float]  # 存储k次生成的时间


class EvaluationSummary(NamedTuple):
    """评估总结结果。"""

    model: str
    model_type: str
    evaluation_method: str
    dataset: str
    k: int  # pass@k中的k值
    total_samples: int
    passed_samples: int
    pass_at_k: float
    evaluation_time: float
    average_time_per_sample: float


# ================================ 自定义异常 ================================


class MBPPEvaluationError(Exception):
    """MBPP评估过程中的基础异常类。"""

    pass


class ModelLoadError(MBPPEvaluationError):
    """模型加载失败异常。"""

    pass


class DatasetLoadError(MBPPEvaluationError):
    """数据集加载失败异常。"""

    pass


class CodeGenerationError(MBPPEvaluationError):
    """代码生成失败异常。"""

    pass


class CodeExecutionTimeoutError(MBPPEvaluationError):
    """代码执行超时异常。"""

    pass


# ================================ 核心功能类 ================================


class ModelInterface:
    """模型接口类，封装模型加载和代码生成功能。"""

    def __init__(self, model_path: str) -> None:
        """初始化模型接口。

        Args:
            model_path: 模型路径或HuggingFace模型名称

        Raises:
            ModelLoadError: 当模型加载失败时抛出
        """
        self.model_path = model_path
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._load_model()

    def _load_model(self) -> None:
        """加载Instruct模型和分词器。

        Raises:
            ModelLoadError: 当模型加载失败时抛出
        """
        logger.info(f"正在加载Instruct模型: {self.model_path}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
            )

            logger.info("✅ Instruct模型加载成功")

        except Exception as e:
            error_msg = f"Instruct模型加载失败: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    @property
    def model(self) -> AutoModelForCausalLM:
        """获取模型实例。"""
        if self._model is None:
            raise ModelLoadError("模型未正确加载")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """获取分词器实例。"""
        if self._tokenizer is None:
            raise ModelLoadError("分词器未正确加载")
        return self._tokenizer

    def generate_code(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
        max_new_tokens: int = 512,
    ) -> str:
        """生成代码。

        Args:
            messages: 消息列表，用于 instruct 模型
            config: 生成配置
            max_new_tokens: 最大生成token数量

        Returns:
            生成的代码字符串

        Raises:
            CodeGenerationError: 当代码生成失败时抛出
        """

        # 对于 instruct 模型，使用 chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Switches between thinking and non-thinking modes.
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 生成代码
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
            )

        # 解码新生成的部分
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        generated = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(
            "\n"
        )

        return generated

    @staticmethod
    def _clean_generated_code(generated_text: str) -> str:
        """从生成文本中提取并清理代码（Instruct模型）。

        Args:
            generated_text: 模型生成的完整文本

        Returns:
            清理后的代码字符串
        """

        code = generated_text.strip()

        # 提取markdown代码块
        markdown_patterns = [r"```python\s*\n(.*?)\n```", r"```\s*\n(.*?)\n```"]
        for pattern in markdown_patterns:
            matches = re.findall(pattern, code, re.DOTALL)
            if matches:
                code = max(matches, key=len).strip()
                break

        # 清理空行/注释/print
        lines = code.split("\n")
        cleaned_lines: list[str] = []
        for line in lines:
            stripped_line = line.strip()
            if not any(stripped_line.startswith(prefix) for prefix in ["print", "#"]):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


class CodeExecutor:
    """代码执行器类，负责安全执行和测试生成的代码。"""

    def __init__(self, timeout_seconds: int = 5) -> None:
        """初始化代码执行器。

        Args:
            timeout_seconds: 代码执行超时时间（秒）
        """
        self.timeout_seconds = timeout_seconds

    def test_code_execution(self, code: str, test_cases: list[str]) -> bool:
        """执行代码测试，设置超时避免死循环。

        Args:
            code: 要测试的代码
            test_cases: 测试用例列表

        Returns:
            True如果所有测试用例通过，否则False

        Raises:
            CodeExecutionTimeoutError: 当代码执行超时时抛出
        """

        def timeout_handler(signum: int, frame: Any) -> None:
            raise CodeExecutionTimeoutError(f"代码执行超时（{self.timeout_seconds}秒）")

        try:
            namespace: dict[str, Any] = {}

            # 设置超时处理
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)

            try:
                # 执行代码
                exec(code, namespace)

                # 执行测试用例
                for i, test_case in enumerate(test_cases):
                    try:
                        exec(test_case, namespace)
                        logger.debug(f"测试用例 {i+1} 通过")
                    except Exception as e:
                        logger.debug(f"测试用例 {i+1} 失败: {e}")
                        logger.debug(f"测试用例内容: {test_case}")
                        return False

                return True

            finally:
                # 恢复原来的信号处理器并取消超时
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        except CodeExecutionTimeoutError:
            logger.debug(f"代码执行超时（{self.timeout_seconds}秒），可能存在死循环")
            return False
        except Exception as e:
            logger.debug(f"测试执行过程中发生未预期错误: {e}")
            return False


class DatasetLoader:
    """数据集加载器类。"""

    @staticmethod
    @lru_cache(maxsize=1)
    def load_mbpp_dataset(max_samples: int | None = None) -> datasets.Dataset:
        """加载MBPP数据集（带缓存）。

        Args:
            max_samples: 最大样本数量，None表示加载全部

        Returns:
            MBPP数据集

        Raises:
            DatasetLoadError: 当数据集加载失败时抛出
        """
        try:
            dataset = load_dataset("mbpp", split="test")
            logger.info(f"✅ MBPP数据集加载成功，共{len(dataset)}个样本")

            if max_samples is not None:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
                logger.info(f"限制评估样本数量为: {max_samples}")

            return dataset

        except Exception as e:
            error_msg = f"MBPP数据集加载失败: {e}"
            logger.error(error_msg)
            raise DatasetLoadError(error_msg) from e


class MBPPEvaluator:
    """MBPP评估器主类。"""

    def __init__(self, config: EvaluationConfig) -> None:
        """初始化评估器。

        Args:
            config: 评估配置
        """
        self.config = config
        self.model_interface = ModelInterface(config.model_path)
        self.code_executor = CodeExecutor(config.timeout_seconds)
        self.dataset_loader = DatasetLoader()

    def evaluate(self) -> EvaluationSummary:
        """执行完整的MBPP评估。

        Returns:
            评估总结结果

        Raises:
            MBPPEvaluationError: 当评估过程中发生错误时抛出
        """
        try:
            # 加载数据集
            dataset = self.dataset_loader.load_mbpp_dataset(self.config.max_samples)

            # 创建输出目录
            output_path = Path(self.config.output_dir)
            output_path.mkdir(exist_ok=True)

            # 开始评估
            logger.info(f"开始MBPP Instruct模型0-shot评估（pass@{self.config.k}）...")
            results: list[EvaluationResult] = []
            total_correct = 0
            start_time = time.time()

            for i, sample in enumerate(tqdm(dataset, desc="MBPP评估")):
                result = self._evaluate_single_sample(sample)
                results.append(result)

                if result.test_passed:
                    total_correct += 1

                # 定期打印进度
                if (i + 1) % 10 == 0:
                    current_pass_rate = total_correct / (i + 1)
                    logger.info(
                        f"进度: {i+1}/{len(dataset)}, 当前pass@{self.config.k}: {current_pass_rate:.3f}"
                    )

            # 计算和保存结果
            return self._save_evaluation_results(
                results, start_time, total_correct, output_path
            )

        except Exception as e:
            if isinstance(e, MBPPEvaluationError):
                raise
            error_msg = f"评估过程中发生未预期错误: {e}"
            logger.error(error_msg)
            raise MBPPEvaluationError(error_msg) from e

    def _evaluate_single_sample(self, sample: dict[str, Any]) -> EvaluationResult:
        """评估单个样本（针对Instruct模型）。

        Args:
            sample: 数据集中的单个样本

        Returns:
            单个样本的评估结果
        """
        # 构造提示
        messages = self._build_prompt(sample)

        # 进行k次代码生成和测试
        generated_codes = []
        test_results = []
        code_lengths = []
        generation_times = []

        for i in range(self.config.k):
            # 生成代码
            generation_start = time.time()
            try:
                # 使用统一的 generate_code 方法
                generated_content = self.model_interface.generate_code(
                    messages, self.config.generation_config, self.config.max_new_tokens
                )

                # 代码清理
                generated_code = self.model_interface._clean_generated_code(
                    generated_content
                )
            except CodeGenerationError as e:
                logger.warning(f"任务{sample['task_id']}第{i+1}次代码生成失败: {e}")
                generated_code = "# 代码生成失败\npass"

            generation_time = time.time() - generation_start

            # 测试代码
            test_passed = self.code_executor.test_code_execution(
                generated_code, sample["test_list"]
            )

            # 存储结果
            generated_codes.append(generated_code)
            test_results.append(test_passed)
            code_lengths.append(len(generated_code))
            generation_times.append(generation_time)

        # 计算pass@k: 如果k次生成中至少有一次通过测试，则该样本通过
        sample_passed = any(test_results)

        return EvaluationResult(
            task_id=sample["task_id"],
            prompt=messages,
            generated_codes=generated_codes,
            test_cases=sample["test_list"],
            test_results=test_results,
            test_passed=sample_passed,
            code_lengths=code_lengths,
            generation_times=generation_times,
        )

    @staticmethod
    def _build_prompt(sample: dict[str, Any]) -> list[dict[str, str]]:
        """构建评估提示。

        Args:
            sample: 数据集样本

        Returns:
            构建好的提示字符串
        """
        code = sample["code"]

        # 提取函数签名
        for line in code.split("\n"):
            if line.strip().startswith("def"):
                signature = line
                break
        else:
            signature = ""

        # 构建对话格式的提示
        system_prompt = "You are a helpful Python coding assistant. When given a task, you must output only the requested function code enclosed in a Python code block, without any tests or commentary."
        user_prompt = f"{sample['text']}\nThe function signature should be: {signature}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def _save_evaluation_results(
        self,
        results: list[EvaluationResult],
        start_time: float,
        total_correct: int,
        output_path: Path,
    ) -> EvaluationSummary:
        """保存评估结果并返回统计信息。

        Args:
            results: 所有评估结果
            start_time: 评估开始时间
            total_correct: 通过的样本数量
            output_path: 输出路径

        Returns:
            评估总结
        """
        total_time = time.time() - start_time
        pass_rate = total_correct / len(results) if results else 0.0

        evaluation_summary = EvaluationSummary(
            model=self.config.model_path,
            model_type="base",
            evaluation_method="0-shot",
            dataset="mbpp_original_base",
            k=self.config.k,
            total_samples=len(results),
            passed_samples=total_correct,
            pass_at_k=pass_rate,
            evaluation_time=total_time,
            average_time_per_sample=total_time / len(results) if results else 0,
        )

        # 转换为字典格式用于保存
        summary_dict = evaluation_summary._asdict()
        results_dict = [result._asdict() for result in results]

        # 保存文件
        results_file = output_path / "mbpp_instruct_evaluation_results.json"
        detailed_file = output_path / "mbpp_instruct_detailed_results.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=2)

        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        # 打印结果摘要
        self._print_evaluation_summary(evaluation_summary, results_file, detailed_file)

        return evaluation_summary

    @staticmethod
    def _print_evaluation_summary(
        summary: EvaluationSummary,
        results_file: Path,
        detailed_file: Path,
    ) -> None:
        """打印评估结果摘要。

        Args:
            summary: 评估总结
            results_file: 结果文件路径
            detailed_file: 详细结果文件路径
        """
        logger.info("=" * 50)
        logger.info(f"MBPP Instruct模型0-shot评估完成（pass@{summary.k}）！")
        logger.info("=" * 50)
        logger.info(f"模型: {summary.model}")
        logger.info(f"模型类型: Instruct (后训练模型)")
        logger.info(f"评估方法: 0-shot pass@{summary.k}")
        logger.info(f"总样本数: {summary.total_samples}")
        logger.info(f"通过样本数: {summary.passed_samples}")
        logger.info(f"Pass@{summary.k}: {summary.pass_at_k:.3f}")
        logger.info(f"评估用时: {summary.evaluation_time:.2f}秒")
        logger.info(f"平均每样本: {summary.average_time_per_sample:.2f}秒")
        logger.info(f"结果文件: {results_file}")
        logger.info(f"详细结果: {detailed_file}")
        logger.info("=" * 50)


# ================================ 命令行接口 ================================


def create_arg_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    Returns:
        配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="MBPP Instruct模型0-shot pass@k评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Instruct模型路径或HuggingFace模型名",
    )

    parser.add_argument(
        "--output-dir",
        default="mbpp_results_instruct_non_thinking",
        help="输出目录",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="最大评估样本数（用于测试，默认评估全部）",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="最大生成新token数量",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="代码执行超时时间（秒）",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="pass@k评估中的k值（默认为1）",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别",
    )

    return parser


def setup_logging(log_level: str) -> None:
    """设置日志配置。

    Args:
        log_level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """主函数。"""
    parser = create_arg_parser()
    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    # 创建评估配置
    config = EvaluationConfig(
        model_path=args.model,
        k=args.k,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_length,
        timeout_seconds=args.timeout,
    )

    try:
        # 创建评估器并运行评估
        evaluator = MBPPEvaluator(config)
        results = evaluator.evaluate()

        # 打印成功信息
        print(f"\n🎉 MBPP Instruct模型0-shot评估成功完成！")
        print(f"📊 Pass@{results.k}: {results.pass_at_k:.3f}")
        print(f"🤖 模型类型: Instruct (后训练模型)")
        print(f"🎯 评估方法: 0-shot pass@{results.k}")
        print(f"📁 结果保存在: {args.output_dir}")

    except MBPPEvaluationError as e:
        logger.error(f"评估失败: {e}")
        print(f"\n❌ MBPP Instruct模型评估失败: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"未预期的错误: {e}")
        print(f"\n❌ 发生未预期的错误: {e}")
        exit(1)


# ================================ 日志配置 ================================

logger = logging.getLogger(__name__)


# ================================ 主程序入口 ================================

if __name__ == "__main__":
    main()
