"""
在ESConv测试集上评测模型性能
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch

# 添加根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.eval.metrics import compute_all_metrics, print_metrics

def load_model(
        model_path: str,
        base_model_path: str = None,
        sft_model_path: str = None,
        device: str = "cuda"
):
    """_summary_

    Args:
        model_path (str): 模型路径，LoRA权重路径
        base_model_path (str, optional): LoRA微调下的基础模型路径. Defaults to None.
        sft_model_path (str, optional): SFT LoRA 路径. Defaults to None.
        device (str, optional): 运行设备. Defaults to "cuda".
    """
    
    model_path = Path(model_path)

    # 检查是否是lora
    is_lora = (model_path / "adapter_config.json").exists()

    if is_lora and base_model_path:
        print("="*50)
        print("加载LoRA模型...")
        print(f"基础模型：{base_model_path}")
        print(f"目标LoRA：{model_path}")
        
        if sft_model_path:
            print(f"SFT LoRA：{sft_model_path}")
        print("="*50)
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 检查是否需要先融合 SFT LoRA （GRPO 评测场景）
        sft_path = Path(sft_model_path) if sft_model_path else None
        target_path = model_path.resolve()
        sft_resolved = sft_path.resolve() if sft_path else None
        
        # 判断是否是GRPO 模式： sft_model_path 存在 且与model_path 不同
        is_grpo_mode = (
            sft_model_path is not None and
            sft_resolved is not None and
            sft_resolved != target_path
        )
        
        if is_grpo_mode:
            print(f"\n[GRPO模式] 先融合 SFT LoRA...")
            print(f"  SFT 路径: {sft_model_path}")
            
            # 加载并融合 SFT LoRA
            base_model = PeftModel.from_pretrained(base_model, sft_model_path)
            base_model = base_model.merge_and_unload()
            print(f"  ✓ SFT LoRA 已融合")
            
            # 清除可能残留的 PEFT 属性
            if hasattr(base_model, 'peft_config'):
                delattr(base_model, 'peft_config')
            if hasattr(base_model, 'active_adapter'):
                delattr(base_model, 'active_adapter')
            
            print(f"  ✓ SFT LoRA 已融合")
            
            # 加载 GRPO LoRA
            print(f"\n[GRPO模式] 加载 GRPO LoRA...")
            print(f"  GRPO 路径: {model_path}")
            model = PeftModel.from_pretrained(base_model, str(model_path))
            print(f"  ✓ GRPO LoRA 已加载")
            
            # 验证 GRPO LoRA 确实被加载
            if hasattr(model, 'peft_config'):
                print(f"\n[验证] PEFT 配置:")
                for adapter_name, config in model.peft_config.items():
                    print(f"  Adapter: {adapter_name}")
                    print(f"    r: {config.r}")
                    print(f"    lora_alpha: {config.lora_alpha}")
                    print(f"    target_modules: {config.target_modules}")
        else:
            # 纯 SFT 模式：直接加载单个 LoRA
            print(f"\n[SFT模式] 直接加载 LoRA...")
            model = PeftModel.from_pretrained(base_model, str(model_path))
            print(f"  ✓ LoRA 已加载: {model_path}")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
    else:
        print(f"加载完整模型：{model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {total_params / 1e9:.2f}B")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")

    return model, tokenizer

def generate_responses(
        model,
        tokenizer,
        dataset,
        max_samples: int = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
):
    """生成回复

    Args:
        model (_type_): 模型
        tokenizer (_type_): 分词器
        dataset (_type_): 数据集
        max_samples (int, optional): 最大样本数. Defaults to None.
        max_new_tokens (int, optional): 最大生成长度. Defaults to 512.
        temperature (float, optional): 采样温度. Defaults to 0.7.
        top_p (float, optional): top-p采样. Defaults to 0.9.
    """

    references = []
    hypotheses = []

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    for sample in tqdm(dataset, desc="生成回复"):
        # 解析messages
        messages = json.loads(sample["messages"])
        target_response = sample["target_response"]

        # 构建输入
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536
        )
        inputs = {
            k: v.to(model.device) for k, v in inputs.items()
        }

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 解码
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        references.append(target_response)
        hypotheses.append(generated_text)
    
    return references, hypotheses

def main():
    parser = argparse.ArgumentParser(description="模型评测")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--base_model_path", type=str, default=None, help="基础模型路径（LoRA模式）")
    parser.add_argument("--sft_model_path", type=str, default=None, help="SFT 阶段的LoRA路径（将在GRPO前被融合）")
    parser.add_argument("--data_path", type=str, default="data/esconv/processed", help="数据集路径")
    parser.add_argument("--split", type=str, default="test", help="数据集划分")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--output_path", type=str, default=None, help="结果保存路径")
    parser.add_argument("--include_bertscore", action="store_true", help="是否计算 BERTScore")

    args = parser.parse_args()

    print("="*60)
    print("ESC 模型评测")
    print("="*60)

    # 加载模型
    model, tokenizer = load_model(args.model_path,
                                  args.base_model_path,
                                  args.sft_model_path
                                  )
    # 加载数据集
    print(f"\n加载数据集：{args.data_path}")
    dataset = load_from_disk(args.data_path)
    eval_dataset = dataset[args.split]
    print(f"评测样本数：{min(args.max_samples, len(eval_dataset)) if args.max_samples else len(eval_dataset)}")

    # 生成回复
    print("\n 生成回复...")
    references, hypotheses = generate_responses(
        model,
        tokenizer,
        eval_dataset,
        max_samples=args.max_samples
    )

    # 计算指标
    print("\n计算评测指标...")
    metrics = compute_all_metrics(
        references,
        hypotheses,
        include_bertscore=args.include_bertscore
    )
    print_metrics(metrics)

    # 保存结果
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "model_path": str(args.model_path),
            "base_model_path": str(args.base_model_path) if args.base_model_path else None,
            "sft_model_path": str(args.sft_model_path) if args.sft_model_path else None,
            "data_path": args.data_path,
            "split": args.split,
            "num_samples": len(references),
            "metrics": metrics
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到：{output_path}")

    # 保存生成样本
    samples_path = Path(args.output_path).parent / "samples.json" if args.output_path else None
    if samples_path:
        samples = []
        for i, (ref, hyp) in enumerate(zip(references[:100], hypotheses[:100])):
            samples.append({
                "index": i,
                "reference": ref,
                "hypothesis": hyp
            })
        
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"生成样本已保存到：{samples_path}")


if __name__ == "__main__":
    main()
