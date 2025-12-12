# ESC Demo：情感支持对话 SFT & GRPO

基于 Qwen3-4B 的情感支持对话微调项目，包含 ESConv 数据的 SFT 以及 ExTES 数据的 GRPO 强化学习范式。

## 目录结构
- configs/sft_esconv.yaml：ESConv SFT 训练配置
- configs/rl_extes_grpo.yaml：ExTES GRPO 配置
- src/data/esconv.py：ESConv 数据解析与 SFT 样本构建
- src/data/extes.py：ExTES 数据解析与 RL 样本构建
- src/training/data_collator.py：SFT 数据整理器（构建 input_ids/labels）
- src/training/sft_trainer.py：LoRA SFT 训练脚本
- scripts/evaluate_model.py：推理/评测脚本
- data/**：原始与处理后的数据目录（示例路径：data/esconv/processed, data/extes/processed）
- outputs/**：模型输出与日志

## 环境准备
```bash
# 建议使用 Python 3.10+ 与 CUDA 12.x
pip install -r requirements.txt
