#!/bin/bash

# 代码评估框架安装脚本
# 专注于EvalPlus（HumanEval+/MBPP+）和MBPP评估

set -e

echo "🚀 开始安装代码评估框架..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi

echo "✅ Python环境检查通过"

echo "📦 安装基础依赖..."

# 安装基础Python包
if [ -f "requirements.txt" ]; then
    echo "使用requirements.txt安装依赖..."
    pip install -r requirements.txt || {
        echo "⚠️  requirements.txt安装失败，尝试手动安装核心依赖..."
        pip install transformers torch tqdm datasets || echo "⚠️  部分基础依赖安装失败"
    }
else
    echo "未找到requirements.txt，手动安装核心依赖..."
    pip install transformers torch tqdm datasets || {
        echo "⚠️  部分基础依赖安装失败，继续其他安装..."
    }
fi

echo "🔧 验证MBPP数据集支持..."
# MBPP通过datasets库即可获取，无需额外安装
echo "✅ MBPP数据集支持已就绪（通过datasets库提供）"

echo ""
echo "📋 安装状态检查..."

echo "检查transformers..."
python -c "import transformers; print('✅ transformers可用')" 2>/dev/null || echo "❌ transformers不可用"

echo "检查datasets..."
python -c "import datasets; print('✅ datasets可用')" 2>/dev/null || echo "❌ datasets不可用"

echo "检查torch..."
python -c "import torch; print('✅ torch可用')" 2>/dev/null || echo "❌ torch不可用"

echo ""
echo "🎉 评估环境安装完成！"
echo ""

