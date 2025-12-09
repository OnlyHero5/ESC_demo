"""
数据处理主脚本
处理 ESConv 和 ExTES 数据集
"""

import sys
from pathlib import Path

#添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.esconv import (
    load_esconv,
    analyze_esconv_dataset,
    print_dataset_stats,
    get_sample_dialog
)

from src.data.extes import (
    load_extes,
    analyze_extes_dataset,
    print_extes_stats,
    get_sample
)

def main():
    print("\n" + "=" * 70)
    print("ESC 数据集处理流水线")
    print("="*70)

    # ================ESConv=================
    print("\n" + "="*70)
    print("Step 1: 处理 ESConv数据集（SFT训练用）")
    print("="*70)


    esconv_raw_path = project_root/ "data" / "esconv" / "raw" / "ESConv.json"

    if not esconv_raw_path.exists():
        print(f" 错误：ESConv数据文件不存在：{esconv_raw_path}")
        print("请确认数据文件位置正确。")
        return
    
    esconv_dataset = load_esconv(
        data_path=str(esconv_raw_path),
        save_processed=True,
        processed_dir=str(project_root/ "data" / "esconv" / "processed")
    )

    # 打印统计信息
    esconv_stats = analyze_esconv_dataset(esconv_dataset)
    print_dataset_stats(esconv_stats)

    # 打印样本信息
    print("\n ESConv 样本示例:")
    sample = get_sample_dialog(esconv_dataset, "train", 0)
    print(f"   对话 ID: {sample['dialog_id']}")
    print(f"   策略: {sample['strategy']}")
    print(f"   目标回复: {sample['target_response'][:100]}...")

    # ==============ExTES=================
    print("\n" + "="*70)
    print("Step 2: 处理 ExTES数据集（RL训练用）")
    print("="*70)

    extes_raw_path = project_root / "data" / "extes" / "raw" / "ExTES.json"

    rl_dataset = load_extes(
        data_path=str(extes_raw_path),
        use_esconv_fallback=False,
        save_processed=True,
        processed_dir=str(project_root / "data" / "extes" / "processed")
    )

    rl_stats = analyze_extes_dataset(rl_dataset)
    print_extes_stats(rl_stats)

    # 打印样本示例
    print("\n RL 样本示例:")
    rl_sample = get_sample(rl_dataset, "train", 0)
    print(f"   对话 ID: {rl_sample['dialog_id']}")
    print(f"   场景: {rl_sample['scene']}")
    print(f"   参考回复: {rl_sample['reference_response'][:100]}...")
    
    # ==================== Step 3: 汇总 ====================
    print("\n" + "=" * 70)
    print("数据处理完成汇总")
    print("=" * 70)
    
    print("\n【SFT 数据集 (ESConv)】")
    print(f"  训练集: {len(esconv_dataset['train'])} 样本")
    print(f"  验证集: {len(esconv_dataset['validation'])} 样本")
    print(f"  测试集: {len(esconv_dataset['test'])} 样本")
    
    print("\n【RL 数据集 (ExTES)】")
    print(f"  训练集: {len(rl_dataset['train'])} 样本")
    print(f"  验证集: {len(rl_dataset['validation'])} 样本")
    
    print("\n【文件位置】")
    print(f"  ESConv 处理后: data/esconv/processed/")
    print(f"  RL 数据处理后: data/extes/processed/")
    
    print("\n" + "=" * 70)
    print("✅ 所有数据处理完成！可以开始 SFT 训练了。")
    print("=" * 70)



if __name__ == "__main__":
    main()
