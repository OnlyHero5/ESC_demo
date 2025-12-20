"""
ESC-Eval 完整评测脚本

流程：
    1. 加载待评测模型
    2. 加载ESC-Role 进行多轮对话
    3. 卸载 ESC-Role, 加载 ESC-RANK进行评分
    4. 汇总并输出结果
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.eval.esc_role import (
    ESCRole,
    ESCModel,
    load_role_cards,
    run_all_dialogues,
    save_dialogues
)
from src.eval.esc_rank import (
    ESCRank,
    compute_average_scores,
    print_esc_rank_results
)

def run_esc_eval(
        config_path: str,
        models_to_eval: list = None,
        max_samples: int = None,
        skip_dialogue: bool = False,
        skip_scoring: bool = False
):
    """运行 ESC-Eval 评测

    Args:
        config_path (str): 配置文件路径
        models_to_eval (list, optional): 要评测的模型名称列表. Defaults to None.表示所有模型
        max_samples (int, optional): 最大样本数. Defaults to None.
        skip_dialogue (bool, optional): 跳过对话生成（使用已有对话文件）. Defaults to False.
        skip_scoring (bool, optional): 跳过评分（只生成对话）. Defaults to False.
    """
    print("="*60)
    print("ESC-Eval 情感支持对话评测")
    print("="*60)

    # 1. 加载配置
    config = load_config(config_path)

    role_cards_path = config["data"]["role_cards_path"]
    output_dir = Path(config["data"]["output_dir"])
    num_turns = config["dialogue"]["num_epochs"]
    language = config["dialogue"]["language"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载角色卡数据
    print(f"\n加载角色卡：{role_cards_path}")
    role_cards = load_role_cards(role_cards_path)

    if max_samples:
        role_cards = role_cards[:max_samples]
        print(f"限制评测样本数：{max_samples}")
    
    # 3.获取待评测模型列表
    all_models = config["models_to_evaluate"]
    if models_to_eval:
        all_models = [m for m in all_models if m["name"] in models_to_eval]

    print(f"\n待评测模型：{[m["name"] for m in all_models]}")

    # 4.对每个模型进行评测
    all_results = {}

    for model_info in all_models:
        model_name = model_info["name"]
        print("\n" + "="*60)
        print(f"评测模型：{model_name} - {model_info['description']}")
        print("="*60)

        dialogue_path = output_dir / f"dialogues_{model_name}_{language}.json"
        score_path = output_dir / f"scores_{model_name}_{language}.json"

        # 4.1 生成对话
        if not skip_dialogue:
            print("\n[Step 1]生成对话...")

            # 加载 ESC-Role
            esc_role = ESCRole(
                model_path=config["esc_role"]["model_path"],
                torch_dtype=config["esc_role"]["torch_dtype"],
                device_map=config["esc_role"]["device_map"],
                max_new_tokens=config["esc_role"]["max_new_tokens"]
            )

            # 加载待评测模型
            esc_model = ESCModel(
                model_path=model_info["model_path"],
                base_model_path=model_info.get("base_model_path"),
                torch_dtype="bfloat16",
                device_map="auto",
                max_new_tokens=512
            )

            # 运行对话
            dialogues = run_all_dialogues(
                esc_model=esc_model,
                esc_role=esc_role,
                role_cards=role_cards,
                num_turns=num_turns,
                language=language
            )
            # 保存对话
            save_dialogues(dialogues, str(dialogue_path))

            # 卸载模型释放内存
            esc_model.unload()
            esc_role.unload()
            torch.cuda.empty_cache()
        
        else:
            print(f"\n[Step 1] 跳过对话生成，使用已有文件：{dialogue_path}")
            with open(dialogue_path, "r", encoding="utf-8") as f:
                dialogues = json.load(f)
        
        # 4.2 评分
        if not skip_scoring:
            print(f"\n[Step 2] 使用 ESC-RANK 评分...")

            # 加载 ESC-RANK
            esc_rank = ESCRank(
                base_model_path=config["esc_rank"]["base_model_path"],
                adapter_path=config["esc_rank"]["adapter_path"],
                torch_dtype=config["esc_rank"]["torch_dtype"],
                language=language
            )

            # 评分
            scores = esc_rank.score_all_dialogues(dialogues, str(score_path))

            # 计算平均分
            avg_scores = compute_average_scores(scores)
            print_esc_rank_results(avg_scores, model_name)

            all_results[model_name] = {
                "description": model_info["description"],
                "scores": avg_scores,
                "num_samples": len(scores)
            }

            # 卸载模型
            esc_rank.unload()
            torch.cuda.empty_cache()
        else:
            print(f"\n[Step 2] 跳过评分")
    
    # 汇总结果
    if all_results:
        print("\n" + "-"*60)
        print("评测结果汇总")
        print("="*60)

        # 打印对比表格
        print_comparison_table(all_results)

        # 保存最终结果
        final_result_path = output_dir / f"esc_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(final_result_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到：{final_result_path}")

        generate_markdown_report(all_results, output_dir / "esc_eval_report.md")

    print("\n" + "="*60)
    print("ESC-Eval 评测完成！")
    print("=" * 60)

    return all_results


def print_comparison_table(results: dict):
    """打印对比表格"""
    dimensions = ["fluency", "diversity", "empathic", "suggestion", "human", "tech", "overall"]
    dim_names = {
        "fluency": "流畅度",
        "diversity": "多样性",
        "empathic": "共情",
        "suggestion": "建议",
        "human": "拟人",
        "tech": "情感知识",
        "overall": "整体"
    }
    
    models = list(results.keys())
    
    # 表头
    header = f"{'维度':<12}" + "".join([f"{m:<12}" for m in models])
    print(header)
    print("-" * len(header))
    
    # 数据行
    for dim in dimensions:
        row = f"{dim_names[dim]:<12}"
        for model in models:
            score = results[model]["scores"].get(dim, 0)
            row += f"{score:<12.2f}"
        print(row)
    
    print("-" * len(header))
    
    # 平均分
    row = f"{'平均':<12}"
    for model in models:
        scores = results[model]["scores"]
        avg = sum(scores.values()) / len(scores) if scores else 0
        row += f"{avg:<12.2f}"
    print(row)


def generate_markdown_report(results: dict, output_path: Path):
    """生成 Markdown 格式报告"""
    dimensions = ["fluency", "diversity", "empathic", "suggestion", "human", "tech", "overall"]
    dim_names = {
        "fluency": "流畅度",
        "diversity": "多样性",
        "empathic": "共情能力",
        "suggestion": "建议有效性",
        "human": "拟人度",
        "tech": "情感知识",
        "overall": "整体偏好"
    }
    
    models = list(results.keys())
    
    report = "# ESC-Eval 评测报告\n\n"
    report += f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## 模型说明\n\n"
    for model, info in results.items():
        report += f"- **{model}**: {info['description']} (样本数: {info['num_samples']})\n"
    
    report += "\n## 评测结果对比\n\n"
    
    # 表格
    report += "| 维度 | " + " | ".join(models) + " |\n"
    report += "|" + "---|" * (len(models) + 1) + "\n"
    
    for dim in dimensions:
        row = f"| {dim_names[dim]} |"
        for model in models:
            score = results[model]["scores"].get(dim, 0)
            row += f" {score:.2f} |"
        report += row + "\n"
    
    # 平均分
    report += "| **平均** |"
    for model in models:
        scores = results[model]["scores"]
        avg = sum(scores.values()) / len(scores) if scores else 0
        report += f" **{avg:.2f}** |"
    report += "\n"
    
    report += "\n## 评分说明\n\n"
    report += "- 分数范围: 0-4 分\n"
    report += "- 0: 非常差\n"
    report += "- 1: 较差\n"
    report += "- 2: 一般\n"
    report += "- 3: 良好\n"
    report += "- 4: 优秀\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Markdown 报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ESC-Eval 评测")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/esc_eval.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="要评测的模型名称，如: baseline sft grpo"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评测样本数（用于快速测试）"
    )
    parser.add_argument(
        "--skip_dialogue",
        action="store_true",
        help="跳过对话生成，使用已有对话文件"
    )
    parser.add_argument(
        "--skip_scoring",
        action="store_true",
        help="跳过评分，只生成对话"
    )
    
    args = parser.parse_args()
    
    run_esc_eval(
        config_path=args.config,
        models_to_eval=args.models,
        max_samples=args.max_samples,
        skip_dialogue=args.skip_dialogue,
        skip_scoring=args.skip_scoring
    )


if __name__ == "__main__":
    main()