"""
GRPO 强化学习脚本

使用 TRL 的 GRPOTrainer 在 ExTES 数据集上进行强化学习
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.rl.reward_functions import create_reward_functions

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s -%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_sft_model_and_tokenizer(config: Dict[str, Any]):
    """
    加载微调之后的模型和分词器
    
    支持 LoRA adapter 和 完整模型模式
    """
    model_config = config["model"]
    sft_model_path = model_config["sft_model_path"]
    base_model_path = model_config["base_model_path"]

    logger.info(f"基础模型路径：{base_model_path}")
    logger.info(f"SFT 模型路径：{sft_model_path}")

    # 优先使用 SFT 路径 加载 Tokenizer
    tokenizer_path = sft_model_path if Path(sft_model_path).exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 数据类型
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(
        model_config.get("torch_dtype", "bfloat16"),
        torch.bfloat16
    )
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": model_config.get("device_map", "auto")
    }
    if model_config.get("use_flash_attention", True):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # 检查是否是LoRA adapter
    adapter_config_path = Path(sft_model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        logger.info("检测到LoRA adapter, 加载基础模型合并...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        model = PeftModel.from_pretrained(model, sft_model_path)
        model = model.merge_and_unload()
        logger.info("LoRA 权重已合并到基础模型")
    else:
        logger.info("加载完整的SFT模型...")
        model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            **model_kwargs
        )
    
    # 梯度检查点
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"模型参数量：{param_count:.2f}B")

    return model, tokenizer

def apply_lora_for_rl(model, config: Dict[str, Any]):
    """为GRPO 训练应用新的LoRA

    在已经合并SFT LoRA的基础上再应用新的LoRA
    """
    lora_config = config.get("lora")

    if not lora_config.get("enabled", True):
        logger.info("LoRA未启用， 使用全参数训练")
        return model

    logger.info("为GRPO 训练应用新的LoRA")
    target_modules = lora_config.get("target_modules", 
                                     [
                                          "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
                                     ])
    
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=list(target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)

    # 打印参数统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA 可训练参数：{trainable / 1e6:.2f}M ({100 * trainable / total: .2f}%)")

    return model

def load_rl_dataset(config: Dict[str, Any], tokenizer) -> Dataset:
    """加载并处理RL 数据集"""
    data_config = config["data"]
    data_path = data_config["train_path"]
    max_prompt_length = data_config.get("max_prompt_length", 1024)

    logger.info(f"加载RL数据集：{data_path}")
    dataset = load_from_disk(data_path)
    train_dataset = dataset["train"]

    logger.info(f"原始样本数：{len(train_dataset)}")

    def process_sample(sample):
        """处理单个样本， 构建prompt 并保留 references"""
        if isinstance(sample["messages"], str):
            messages = json.loads(sample["messages"])
        else:
            messages = sample["messages"]
        
        # 使用 chat template 构建 prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 截断处理
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_length:
            logger.warning(
                f"Prompt 长度 {len(prompt_ids)} 超过限制 {max_prompt_length} 将被截断"
            )
            # 保留后半部分
            prompt_ids = prompt_ids[-max_prompt_length:]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        
        return {
            "prompt": prompt,
            "references": sample["reference_response"]
        }

    processed_dataset = train_dataset.map(
        process_sample,
        remove_columns=train_dataset.column_names,
        desc="格式化RL 数据集"
    )

    logger.info(f"处理后的样本：{len(processed_dataset)}")
    logger.info(f"数据集列名：{processed_dataset.column_names}")

    return processed_dataset

def create_grpo_config(config: Dict[str, Any]) -> GRPOConfig:
    """创建GRPO训练配置"""
    train_config = config["training"]
    grpo_config = config["grpo"]
    log_config = config.get("logging", {})
    data_config = config["data"]

    output_dir = train_config["output_dir"]
    tensorboard_dir = log_config.get("tensorboard_dir", f"{output_dir}/tensorboard")

    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    return GRPOConfig(
        output_dir=output_dir,

        # GRPO核心参数
        num_generations=grpo_config.get("num_generations", 4),
        temperature=grpo_config.get("temperature", 0.7),
        max_prompt_length=data_config.get("max_prompt_length", 1024),
        max_completion_length=data_config.get("max_completion_length", 256),
        beta=grpo_config.get("beta", 0.04),
        
        # 训练参数
        num_train_epochs=train_config.get("num_epochs", 1),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),

        # 学习率
        learning_rate=train_config.get("learning_rate", 1e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),

        # 日志
        logging_steps=train_config.get("logging_steps", 10),
        logging_dir=tensorboard_dir,

        # 保存
        save_strategy="steps",
        save_steps=train_config.get("save_steps", 200),
        save_total_limit=train_config.get("save_total_limit", 3),

        # 精度
        bf16=train_config.get("bf16", True),
        fp16=False,

        # 其他
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        seed=train_config.get("seed", 42),
        max_steps=train_config.get("max_steps", -1),
        remove_unused_columns=False,

        # 报告
        report_to=["tensorboard"] if log_config.get("use_tensorboard", True) else [],
    )

def run_grpo_training(config_path: str):
    """
    运行GRPO 强化学习
    
    数据流：Dataset(prompt, references) -> GRPOTrainer -> reward_funcs(completions, references)
    """
    
    logger.info("="*60)
    logger.info("GRPO 强化学习训练")
    logger.info("="*60)

    # 1.加载配置
    config = load_config(config_path)
    logger.info(f"配置文件：{config_path}")

    # 2.设置随机种子
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"随机种子：{seed}")

    # 3.加载模型和分词器
    logger.info("="*40)
    logger.info("加载模型...")
    logger.info("="*40)
    model, tokenizer = load_sft_model_and_tokenizer(config)

    # 4. 应用LoRA
    logger.info("="*40)
    logger.info("配置LoRA...")
    model = apply_lora_for_rl(model, config)

    # 5. 加载数据集
    logger.info("="*40)
    logger.info("加载数据集...")
    train_dataset = load_rl_dataset(config, tokenizer)

    # 6. 创建奖励函数
    logger.info("="*40)
    logger.info("初始化奖励函数")
    reward_funcs = create_reward_functions(config)
    logger.info(f"奖励函数数量：{len(reward_funcs)}")

    # 7. 创建训练配置
    logger.info("="*40)
    logger.info("创建训练配置")
    grpo_config = create_grpo_config(config)

    # 打印关键配置
    logger.info(f"  num_generations: {grpo_config.num_generations}")
    logger.info(f"  temperature: {grpo_config.temperature}")
    logger.info(f"  beta (KL): {grpo_config.beta}")
    logger.info(f"  learning_rate: {grpo_config.learning_rate}")
    logger.info(f"  batch_size: {grpo_config.per_device_train_batch_size}")
    logger.info(f"  gradient_accumulation: {grpo_config.gradient_accumulation_steps}")

    # 8. 创建Trainer
    logger.info("="*40)
    logger.info("创建GRPOTrainer...")
    logger.info("="*40)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs
    )

    # 9.开始训练
    logger.info("="*60)
    logger.info("开始GRPO训练...")
    logger.info("="*60)

    trainer.train()

    # 10.保存最终模型
    final_output_dir = os.path.join(config["training"]["output_dir"], "final")
    os.makedirs(final_output_dir, exist_ok=True)

    logger.info(f"保存最终模型到：{final_output_dir}")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # 保存训练配置
    config_save_path = os.path.join(final_output_dir, "training_config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info("GRPO 训练完成!")
    logger.info(f"模型保存位置: {final_output_dir}")
    logger.info("=" * 60)
    
    return trainer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO 强化学习训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl_extes_grpo.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    run_grpo_training(args.config)


