# LLama2_self_defined

本项目为 LLama2 模型的简化自定义实现，涵盖了核心模块如 Attention、MLP、RMSNorm、Rotary Embedding 等，旨在帮助理解大模型的基本结构与实现细节。

## 目录结构

```
LLama2_self_difined/
├── attention.py         # 注意力机制实现
├── decoder.py           # 解码器层实现
├── mlp.py               # 前馈神经网络模块
├── model.py             # 模型主文件（待完善）
├── module/
│   ├── RMSnorm.py       # RMSNorm归一化层
│   ├── config.py        # 模型配置
│   ├── repeat_kv.py     # KV重复工具
│   ├── rotary_emb.py    # 旋转位置编码
├── requirement.txt      # 依赖库
├── README               # 项目说明
```

## 依赖环境

请确保已安装以下依赖（可通过 `pip install -r requirement.txt` 安装）：

- transformers
- torch
- typing

## 快速开始

以 `mlp.py` 和 `attention.py` 为例，均内置了测试函数，可直接运行：

```bash
python mlp.py
python attention.py
```

## 主要模块说明

- **attention.py**：实现了多头自注意力机制，支持 Flash Attention。
- **mlp.py**：实现了带SILU激活的前馈神经网络。
- **module/RMSnorm.py**：实现了 RMSNorm 归一化层。
- **module/rotary_emb.py**：实现了旋转位置编码（RoPE）。
- **module/config.py**：集中管理模型超参数配置。
- **module/repeat_kv.py**：用于多头注意力中键值的重复扩展。

## 参考教程

本项目参考自 [Datawhale Happy LLM 教程](https://datawhalechina.github.io/happy-llm/#/)，欢迎学习与交流。

---
如有问题或建议，欢迎 issue 或 PR！
