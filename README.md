# LLama2_self_defined

本项目为 LLama2 模型的简化自定义实现，涵盖了核心模块如 Attention、MLP、RMSNorm、Rotary Embedding、Tokenizer 等，旨在帮助理解大模型的基本结构与实现细节，并支持自定义数据集和分词器训练。

## 目录结构

```
LLama2_self_difined/
├── model/
│   ├── attention.py     # 注意力机制实现
│   ├── decoder.py       # 解码器层实现
│   ├── mlp.py           # 前馈神经网络模块
│   ├── transformer.py   # Transformer主模型
│   ├── module/
│   │   ├── RMSnorm.py   # RMSNorm归一化层
│   │   ├── config.py    # 模型配置
│   │   ├── repeat_kv.py # KV重复工具
│   │   ├── rotary_emb.py# 旋转位置编码
├── tokenizer/
│   ├── tokenizer.py     # 分词器训练与评估
├── utils/
│   ├── download_dataset.py # 数据集下载与预处理
├── requirement.txt      # 依赖库
├── README.md            # 项目说明
```

## 依赖环境

请确保已安装以下依赖（可通过 `pip install -r requirement.txt` 安装）：

- transformers
- torch
- typing
- tokenizers
- datasets
- tqdm
- modelscope
- huggingface_hub

## 快速开始

### 1. 训练Tokenizer

自定义分词器训练与评估：

```bash
cd tokenizer
python tokenizer.py
```
请在 `tokenizer.py` 中设置好 `data_path` 路径。

### 2. 数据集下载与预处理

自动下载和处理预训练/SFT数据集：

```bash
cd utils
python download_dataset.py
```
如遇命令行工具问题，请参考代码注释手动下载或调整命令。

### 3. 模型各模块测试

以 `mlp.py`、`attention.py` 为例，均内置了测试函数，可直接运行：

```bash
cd model
python mlp.py
python attention.py
```

### 4. Transformer主模型测试

```bash
cd model
python transformer.py
```

## 主要模块说明

- **model/attention.py**：实现了多头自注意力机制，支持 Flash Attention。
- **model/mlp.py**：实现了带SILU激活的前馈神经网络。
- **model/module/RMSnorm.py**：实现了 RMSNorm 归一化层。
- **model/module/rotary_emb.py**：实现了旋转位置编码（RoPE）。
- **model/module/config.py**：集中管理模型超参数配置。
- **model/module/repeat_kv.py**：用于多头注意力中键值的重复扩展。
- **tokenizer/tokenizer.py**：自定义分词器训练、配置与评估。
- **utils/download_dataset.py**：数据集下载与格式转换脚本。

## 参考教程

本项目参考自 [Datawhale Happy LLM 教程](https://datawhalechina.github.io/happy-llm/#/)，欢迎学习与交流。

---

如有问题或建议，欢迎 issue 或 PR！
