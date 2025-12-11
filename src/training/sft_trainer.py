"""
ESConv SFT 训练脚本

使用LoRA 在 ESConv 数据集上 微调 Qwen3 模型
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    set_seed
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.data_collator import create_sft_data_collator
from src.utils.config import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



class LoggingCallback(TrainerCallback):
    """自定义日志回调"""

    def on_log(
            self,
            args,
            state,
            control,
            logs=None,
            **kwargs,
    ):
        if logs:
            # 只打印重要指标
            important_keys = ["loss", "eval_loss", "learning_rate", "epoch"]
            filtered_logs = {k: v for k, v in logs.items() if k in important_keys}
            if filtered_logs:
                logger.info(f"Step {state.global_step}: {filtered_logs}")



def load_model_and_tokenizer(config: Dict[str, Any]):
    """加载模型和分词器

    Args:
        config (Dict[str, Any]): 配置字典
    
    Returns:
        (model, tokenizer)
    """

    model_config = config["model"]
    model_path = model_config["path"]

    logger.info(f"加载模型：{model_path}")

    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_config.get("trust_remote_code", True),
        padding_side = "left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 设置 数据类型
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(model_config.get("torch_dtype", "bfloat16"), torch.bfloat16)

    # 加载模型
    model_kwargs = {
        "trust_remote_code" : model_config.get("trust_remote_code", True),
        "torch_dtype": torch_dtype,
        "device_map": model_config.get("device_map", "auto")
    }
    if model_config.get("use_flash_attention", True):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )

    # 启用梯度检查点
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    logger.info(f"模型参数量：{model.num_parameters() / 1e9:.2f}")

    return model, tokenizer


def apply_lora(model, config: Dict[str, Any]):
    """应用LoRA配置

    Args:
        model (_type_): 基础模型
        config (Dict[str, Any]): 配置字典
    
    Returns:
        应用LoRA后的模型
    """
    lora_config = config["lora"]

    if not lora_config.get("enable", True):
        logger.info("LoRA未启用，使用全参数微调")
        return model
    
    logger.info("应用LoRA配置...")

    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        }),
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)

    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA可训练参数：{trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

    return model

def load_datasets(config:Dict[str, Any]):
    """加载已经预处理好的数据集"""
    data_config = config["data"]
    data_path = data_config["train_path"]
    
    logger.info(f"加载数据集：{data_path}")

    dataset = load_from_disk(data_path)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    logger.info(f"训练集：{len(train_dataset)} 样本")
    logger.info(f"验证集：{len(eval_dataset)} 样本")

    return train_dataset, eval_dataset

def create_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """创建训练参数

    Args:
        config (Dict[str, Any]): 配置字典

    Returns:
        TrainingArguments: 训练参数对象
    """
    train_config = config["training"]
    log_config = config.get("logging", [])

    # 创建输出目录
    output_dir = train_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        # 训练轮数和批次
        num_train_epochs=train_config.get("num_epochs", 3),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),

        # 学习率
        learning_rate=train_config.get("learning_rate", 5e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),

        # 优化器
        optim=train_config.get("optim", "adamw_torch"),
        adam_beta1=train_config.get("adam_beta1", 0.9),
        adam_beta2=train_config.get("adam_beta2", 0.999),
        adam_epsilon=train_config.get("adam_epsilon", 1e-8),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),

        # 日志
        logging_steps=train_config.get("logging_steps", 10),
        logging_dir=log_config.get("tensorboard_dir", f"{output_dir}/tensorboard"),

        # 保存
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 500),
        save_total_limit=train_config.get("save_total_limit", 3),
        eval_strategy=train_config.get("eval_strategy", "steps"),
        eval_steps=train_config.get("eval_steps", 500),

        # 精度
        bf16=train_config.get("bf16", True),
        fp16=train_config.get("fp16", False),

        # 其他
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        dataloader_num_workers=train_config.get("dataloader_num_workers", 4),
        remove_unused_columns=train_config.get("remove_unused_columns", False),
        seed=train_config.get("seed", 42),

        # 报告
        report_to=["tensorboard"] if log_config.get("use_tensorboard", True) else [],

        # 加载最佳模型
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    return args

def run_sft_training(config_path: str):
    """
    运行 SFT 训练
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    logger.info("=" * 60)
    logger.info("ESConv SFT 训练")
    logger.info("=" * 60)

    config = load_config(config_path)

    # 设置随机种子
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"随机种子：{seed}")

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(config)

    # 应用LoRA
    model = apply_lora(model, config)

    # 加载数据集
    train_dataset, eval_dataset = load_datasets(config)

    # 创建数据整理器
    data_config = config["data"]
    data_collator = create_sft_data_collator(
        tokenizer=tokenizer,
        max_seq_length=data_config.get("max_seq_length", 2048),
        max_prompt_length=data_config.get("max_prompt_length", 1536),
        max_response_lenth=data_config.get("max_response_length", 512)
    )

    # 创建训练参数
    train_args = create_training_arguments(config)

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[LoggingCallback()]
    )

    # 开始训练
    logger.info("="*60)
    logger.info("开始训练")
    logger.info("="*60)

    trainer.train()

    # 保存最终模型
    final_output_dir = os.path.join(config["training"]["output_dir"], "final")
    logger.info(f"保存最终模型到：{final_output_dir}")

    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # 保存训练配置
    config_save_path = os.path.join(final_output_dir, "training_config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info("="*60)
    logger.info("训练完成")
    logger.info("="*60)

    return trainer



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESConv SFT 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_esconv.yaml",
        help="配置文件路径"
    )

    args = parser.parse_args()
    
    run_sft_training(args.config)