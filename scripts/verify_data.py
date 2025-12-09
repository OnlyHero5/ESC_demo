"""
验证处理效果
"""
"""
验证数据处理结果
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_from_disk


def main():
    print("\n" + "=" * 60)
    print("数据处理结果验证")
    print("=" * 60)
    
    # 检查 ESConv 处理结果
    esconv_path = project_root / "data" / "esconv" / "processed"
    print(f"\n【ESConv 数据集】")
    
    if esconv_path.exists():
        esconv = load_from_disk(str(esconv_path))
        print(f"  ✅ 训练集: {len(esconv['train'])} 样本")
        print(f"  ✅ 验证集: {len(esconv['validation'])} 样本")
        print(f"  ✅ 测试集: {len(esconv['test'])} 样本")
        
        # 打印一个样本
        sample = esconv['train'][0]
        messages = json.loads(sample['messages'])
        print(f"\n  📝 样本示例:")
        print(f"     对话 ID: {sample['dialog_id']}")
        print(f"     策略: {sample['strategy']}")
        print(f"     情感类型: {sample['emotion_type']}")
        print(f"     消息数: {len(messages)}")
        print(f"     目标回复: {sample['target_response'][:80]}...")
        
        # 验证数据完整性
        print(f"\n  🔍 数据完整性检查:")
        empty_responses = sum(1 for s in esconv['train'] if not s['target_response'].strip())
        print(f"     空回复样本: {empty_responses}")
        
        strategies = set(s['strategy'] for s in esconv['train'] if s['strategy'])
        print(f"     策略种类: {len(strategies)}")
    else:
        print(f"  ❌ 未找到处理后的数据: {esconv_path}")
    
    # 检查 RL 数据处理结果
    rl_path = project_root / "data" / "extes" / "processed"
    print(f"\n【RL 数据集】")
    
    if rl_path.exists():
        rl_data = load_from_disk(str(rl_path))
        print(f"  ✅ 训练集: {len(rl_data['train'])} 样本")
        print(f"  ✅ 验证集: {len(rl_data['validation'])} 样本")
        
        # 打印一个样本
        sample = rl_data['train'][0]
        messages = json.loads(sample['messages'])
        print(f"\n  📝 RL 样本示例:")
        print(f"     对话 ID: {sample['dialog_id']}")
        print(f"     场景: {sample['scene']}")
        print(f"     上下文轮次: {sample['context_turns']}")
        print(f"     策略: {sample['strategy']}")
        print(f"     参考回复: {sample['reference_response'][:80]}...")
        
        # 验证数据完整性
        print(f"\n  🔍 数据完整性检查:")
        empty_responses = sum(1 for s in rl_data['train'] if not s['reference_response'].strip())
        print(f"     空回复样本: {empty_responses}")
        
        avg_context = sum(s['context_turns'] for s in rl_data['train']) / len(rl_data['train'])
        print(f"     平均上下文轮次: {avg_context:.1f}")
    else:
        print(f"  ❌ 未找到 RL 数据: {rl_path}")
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()