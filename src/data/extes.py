"""
ExTES数据集处理模块
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, DatasetDict
from tqdm import tqdm

EXTES_STRATEGY_MAP = {
    "Reflective Statements": "反思性陈述",
    "Clarification": "澄清",
    "Emotional Validation": "情感验证",
    "Empathetic Statements": "共情陈述",
    "Affirmation": "肯定",
    "Offer Hope": "给予希望",
    "Avoid Judgment and Criticism": "避免评判",
    "Suggest Options": "建议选项",
    "Collaborative Planning": "协作规划",
    "Provide Different Perspectives": "提供不同视角",
    "Reframe Negative Thoughts": "重构负面想法",
    "Share Information": "分享信息",
    "Normalize Experiences": "正常化体验",
    "Promote Self-Care Practices": "促进自我关怀",
    "Stress Management": "压力管理",
    "Others": "其他",
}

def load_extes_raw(data_path: str) -> List[Dict]:
    """加载ExTES 原始JSON数据

    Args:
        data_path (str): 路径

    Returns:
        List[Dict]: 对话列表
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("="*60)
    print(f"加载了{len(data)} 个对话")
    print("="*60)
    return data


def _clean_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("AI", "Response", "response", "text", "content"):
            v = value.get(key)
            if isinstance(v, str):
                return v.strip()
    return ""

def _clean_strategy(item, ai_value):
    for source in (ai_value if isinstance(ai_value, dict) else {}, item):
        if not isinstance(source, dict):
            continue
        for key in ("AI strategy", "AI Strategy", "strategy"):
            v = source.get(key)
            if isinstance(v, str):
                return v.strip()
    return ""

def parse_extes_dialog(dialog: Dict, index: int) -> Dict[str, Any]:
    """解析单个对话

    Args:
        dialog (Dict): 原始对话数据
        index (int): 对话的索引值

    Returns:
        Dict[str, Any]: 标准化的对话数据
    """
    dialog_id = f"extes_{index:04d}"

    # 提取场景信息
    scene = dialog.get("scene", "")
    description = dialog.get("description", "")

    # 解析对话内容
    turns = []
    content = dialog.get("content", "")

    for item in content:
        if "User" in item:
            #用户信息
            text = _clean_text(item["User"])
            if text:
                turns.append({
                    "speaker": "seeker",
                    "text": text,
                    "strategy": ""
                })
        elif "AI" in item:
            #AI回复
            ai_val = item["AI"]
            text = _clean_text(ai_val)
            strategy = _clean_strategy(item, ai_val)
            if text:
                turns.append({
                    "speaker": "supporter",
                    "text": text,
                    "strategy": strategy
                })
    
    return {
        "dialog_id": dialog_id,
        "scene": scene,
        "description": description,
        "turns": turns,
        "num_turns": len(turns)
    }



def build_rl_samples(
        dialogs: List[Dict[str, Any]],
        system_prompt: str,
        min_context_turns: int = 1,
        max_context_turns: int = 40,
        include_description: bool = True,
) -> List[Dict[str, Any]]:
    """构建RL 训练样本

    每个样本包括：
    - prompt: 到当前轮为止的对话历史（用于模型生成）
    - reference_response: 真实的supporter回复（用于计算奖励）

    Args:
        dialogs (List[Dict[str, Any]]): 解析后的对话列表
        system_prompt (str): 系统提示词
        min_context_turns (int, optional): 最小上下文轮次. Defaults to 1.
        max_context_turns (int, optional): 最大上下文轮次. Defaults to 40.
        include_description (bool, optional): _是否包含场景描述. Defaults to True.

    Returns:
        List[Dict[str, Any]]: RL训练样本
    """
    all_samples = []

    for dialog in tqdm(dialogs, desc="构建RL样本"):
        turns = dialog["turns"]

        #构建系统提示
        if include_description and dialog["description"]:
            full_system_prompt = f"{system_prompt}\n\n 【用户情景】{dialog['description']}"
        else:
            full_system_prompt = system_prompt
        
        # 为每个 supporter回复创建RL样本
        history = []

        for i, turn in enumerate(turns):
            if turn["speaker"] == "supporter":
                # 检查是否有足够的上下文
                seeker_count = sum(1 for h in history if h["speaker"] == "seeker")

                if seeker_count >= min_context_turns:
                    # 构建 context消息 （限制最大轮次）
                    context_history = history[-max_context_turns:]

                    # 构建消息列表
                    messages = [
                        {
                            "role": "system",
                            "content": full_system_prompt,
                        }
                    ]

                    for h in context_history:
                        role = "user" if h["speaker"] == "seeker" else "assistant"
                        messages.append({
                            "role": role,
                            "content": h["text"]
                        })

                    all_samples.append({
                        "dialog_id": dialog["dialog_id"],
                        "turn_index": i,
                        "messages": messages,
                        "reference_response": turn["text"],
                        "strategy": turn["strategy"],
                        "scene": dialog["scene"],
                        "context_turns": len(context_history)
                    })
            history.append(turn)
        
    print("="*60)
    print(f"共生成 {len(all_samples)} 个 RL训练样本")
    print("="*60)
    return all_samples


# =====备用方案=====
def build_rl_samples_from_esconv(
        esconv_path: str,
        system_prompt: str,
        min_context_turns: int = 1
) -> List[Dict[str, Any]]:
    """从ESConv处理后的数据构建RL样本(替代方案)

    Args:
        esconv_path (str): ESConv processed数据
        system_prompt (str): 系统提示词
        min_context_turns (int, optional): 最小上下文轮次. Defaults to 1.

    Returns:
        List[Dict[str, Any]]: RL 样本
    """
    from datasets import load_from_disk

    esconv_dataset = load_from_disk(esconv_path)
    all_samples = []

    for split in ["train", "validation"]:
        for sample in tqdm(esconv_dataset[split], desc=f"从 ESConv {split} 构建RL样本"):
            messages = json.loads(sample["messages"])

            # 过滤：至少需要 system + min_context_turns 个用户消息
            user_count = sum(1 for m in messages if m["role"] == "user")

            if user_count >= min_context_turns:
                all_samples.append({
                    "dialog_id": sample["dialog_id"],
                    "turn_index": sample["turn_index"],
                    "messages": messages,
                    "reference_response": sample["target_response"],
                    "strategy": sample["strategy"],
                    "scene": sample.get("problem_type", ""),
                    "context_turns": len(messages) -1
                })
    
    print("="*60)
    print(f" 从 ESConv 生成 {len(all_samples)} 个 RL 训练样本")
    print("="*60)
    return all_samples



def split_rl_dataset(
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    划分RL数据集
    """
    import random
    random.seed(seed)

    # 按对话ID划分
    dialog_samples = {}
    for sample in samples:
        dialog_id = sample["dialog_id"]
        if dialog_id not in dialog_samples:
            dialog_samples[dialog_id] = []
        dialog_samples[dialog_id].append(sample)
    
    dialog_ids = list(dialog_samples.keys())
    random.shuffle(dialog_ids)

    n_train = int(len(dialog_ids) * train_ratio)
    train_ids = dialog_ids[:n_train]
    val_ids = dialog_ids[n_train:]

    train_samples = [s for did in train_ids for s in dialog_samples[did]]
    val_samples = [s for did in val_ids for s in dialog_samples[did]]

    print("="*60)
    print(f" RL数据集划分：")
    print(f" 训练集：{len(train_samples)}样本 ({len(train_ids)} 对话)")
    print(f" 验证集：{len(val_samples)}样本 ({len(val_ids)} 对话)")

    return train_samples, val_samples



def create_hf_dataset(
        samples: List[Dict[str, Any]]
) -> Dataset:
    """转换成huggingface dataset格式"""
    processed_samples = []
    for sample in samples:
        processed_samples.append(
            {
                "dialog_id": sample["dialog_id"],
                "turn_index": sample["turn_index"],
                "messages": json.dumps(sample["messages"], ensure_ascii=False),
                "reference_response": sample["reference_response"],
                "strategy": sample["strategy"],
                "scene": sample.get("scene", ""),
                "context_turns": sample["context_turns"]
            }
        )
    return Dataset.from_list(processed_samples)



def load_extes(
        data_path: str = "data/extes/raw/ExTES.json",
        system_prompt: Optional[str] = None,
        include_description: bool = True,
        use_esconv_fallback: bool = True,
        esconv_processed: bool = "data/esconv/processed",
        save_processed: bool = True,
        processed_dir: str = "data/extes/processed"
) -> DatasetDict:
    """加载并处理 ExTES 数据集

    Args:
        data_path (str, optional): 原始数据集. Defaults to "data/extes/raw/ExTES.json".
        system_prompt (Optional[str], optional): 系统提示词. Defaults to None.
        include_description (bool, optional): 是否包含场景描述. Defaults to True.
        use_esconv_fallback (bool, optional): 当ExTES不可用时是否使用ESConv. Defaults to True.
        esconv_processed (bool, optional): 已经预处理ESConv数据集位置. Defaults to "data/esconv/processed".
        save_processed (bool, optional): 是否保存处理后的数据集. Defaults to True.
        processed_dir (str, optional): 保存处理后的数据集的位置. Defaults to "data/extes/processed".

    Returns:
        DatasetDict: huggingface数据集
    """
    if system_prompt is None:
        system_prompt = """
    你是一个专业的情感支持助手。你的目标是：
    1. 认真倾听用户的困扰和情绪
    2. 表达真诚的理解和共情
    3. 使用恰当的情感支持策略（如提问、释义、肯定、建议等）
    4. 帮助用户探索问题、舒缓情绪、找到解决方向

    请用温暖、专业、真诚的方式与用户交流。
    """
    
    print("="*60)
    print("ExTES数据集处理")
    print("="*60)

    all_samples = []
    data_source = "unknown"

    extes_path = Path(data_path)
    if extes_path.exists():
        print(f"\n 加载 ExTES 数据集 ：{extes_path}")

        try:
            raw_data = load_extes_raw(extes_path)

            # 解析对话
            print("\n 解析对话...")
            parsed_dialogs = []
            for idx, d in enumerate(tqdm(raw_data, desc="解析对话")):
                parsed = parse_extes_dialog(d, idx)
                parsed_dialogs.append(parsed)
            
            # 过滤有效对话
            valid_dialogs = [d for d in parsed_dialogs if d["num_turns"] >= 2]
            print(f"    有效对话：{len(valid_dialogs)} / {len(parsed_dialogs)}")

            if valid_dialogs:
                # 构建 RL 样本
                print("\n 构建RL样本...")

                all_samples = build_rl_samples(
                    valid_dialogs,
                    system_prompt,
                    include_description=include_description
                )
                data_source = "ExTES"

        except Exception as e:
            print(f"\n ExTES数据文件加载失败：{e}")
    
    else:
        print(f"ExTES数据文件不存在：{extes_path}")
    
    # 针对ExTES的异常处理
    # 如果 ExTES 样本不足，使用 ESConv 补充
    if len(all_samples) < 100 and use_esconv_fallback:
        print(f"\n 使用 ESConv 作为 RL 数据源...")
        try:
            esconv_samples = build_rl_samples_from_esconv(esconv_processed, system_prompt)
            if not all_samples:
                all_samples = esconv_samples
                data_source = "ESConv"
            else:
                all_samples.extend(esconv_samples)
                data_source = "ExTES+ESConv"
        except Exception as e:
            print(f" ESConv 加载失败: {e}")
    
    if not all_samples:
        raise RuntimeError("没有可用的RL训练数据集！请检查数据文件。")
    
    print(f"\n 数据来源：{data_source}")

    # 划分数据集
    print("\n 划分数据集...")
    train_samples, val_samples = split_rl_dataset(all_samples)

    # 创建 DatasetDict
    dataset_dict = DatasetDict({
        "train": create_hf_dataset(train_samples),
        "validation": create_hf_dataset(val_samples)
    })

    # 保存处理后的数据集
    if save_processed:
        processed_path = Path(processed_dir)
        processed_path.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(processed_path))
        print(f"\n 数据已保存到：{processed_path}")
    
    print("\n" + "="*60)
    print(" RL 数据集处理完成！")
    print("=" * 60)

    return dataset_dict



def load_processed_extes(
        processed_dir: str = "data/extes/processed"
) -> DatasetDict:
    """加载已处理的ExTES 数据集"""
    from datasets import load_from_disk

    processed_path = Path(processed_dir)
    if not processed_path.exists():
        raise FileNotFoundError(f"处理后的数据不存在：{processed_dir}")
    
    dataset = load_from_disk(str(processed_path))
    print(f"\n  已加载RL数据集")
    print(f"    训练集：{len(dataset['train'])} 样本")
    print(f"    验证集：{len(dataset['validation'])}")

    return dataset



def get_sample(
        dataset: DatasetDict,
        split: str = "train",
        index: int = 0
) -> Dict:
    """获取一个样本示例"""
    sample = dict(dataset[split][index])
    sample["messages"] = json.loads(sample["messages"])
    return sample



# ==========统计分析函数==========

def analyze_extes_dataset(
        dataset: DatasetDict
) -> Dict[str, Any]:
    """分析统计情况"""
    stats = {
        "splits": {},
        "strategies": {},
        "scenes": {},
        "context_turns": [],
        "response_lengths": []
    }

    for split_name in dataset.keys():
        split_data = dataset[split_name]
        stats["splits"][split_name] = len(split_data)

        for sample in split_data:
            # 统计策略
            strategy = sample["strategy"]
            if strategy:
                stats["strategies"][strategy] = stats["strategies"].get(strategy, 0) + 1
            
            # 统计场景
            scene = sample.get("scene", "")
            if scene:
                stats["scenes"][scene] = stats["scenes"].get(scene, 0) + 1
            
            # 统计上下文
            stats["context_turns"].append(sample["context_turns"])

            stats["response_lengths"].append(len(sample["reference_response"]))
    
    if stats["context_turns"]:
        stats["avg_context_turns"] = sum(stats["context_turns"]) / len(stats["context_turns"])

    if stats["response_lengths"]:
        stats["avg_response_length"] = sum(stats["response_lengths"]) / len(stats["response_lengths"])

    return stats



def print_extes_stats(stats: Dict[str, Any]):
    """打印统计数据"""
    print("\n" + "=" * 60)
    print(" ExTES/RL 数据集统计")
    print("=" * 60)
    
    print("\n【数据划分】")
    for split, count in stats["splits"].items():
        print(f"  {split}: {count} 样本")
    
    print(f"\n【平均统计】")
    print(f"  平均上下文轮次: {stats.get('avg_context_turns', 0):.1f}")
    print(f"  平均回复长度: {stats.get('avg_response_length', 0):.1f} 字符")
    
    if stats["strategies"]:
        print("\n【策略分布 (Top 10)】")
        sorted_strategies = sorted(stats["strategies"].items(), key=lambda x: x[1], reverse=True)
        total = sum(stats["strategies"].values())
        for strategy, count in sorted_strategies[:10]:
            pct = 100 * count / total
            strategy_cn = EXTES_STRATEGY_MAP.get(strategy, strategy)
            print(f"  {strategy_cn}: {count} ({pct:.1f}%)")
    
    if stats["scenes"]:
        print("\n【场景分布 (Top 10)】")
        sorted_scenes = sorted(stats["scenes"].items(), key=lambda x: x[1], reverse=True)
        for scene, count in sorted_scenes[:10]:
            print(f"  {scene}: {count}")



if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    #处理RL数据集
    dataset = load_extes(
        data_path="data/extes/raw/ExTES.json",
        use_esconv_fallback=False,
        save_processed=True
    )

    # 打印统计信息
    stats = analyze_extes_dataset(dataset)
    print_extes_stats(stats)
    
    # 打印示例
    print("\n" + "=" * 60)
    print("📝 RL 样本示例")
    print("=" * 60)
    
    sample = get_sample(dataset, "train", 0)
    print(f"\n对话 ID: {sample['dialog_id']}")
    print(f"场景: {sample['scene']}")
    print(f"上下文轮次: {sample['context_turns']}")
    print(f"策略: {sample['strategy']}")
    print(f"\n历史消息:")
    for msg in sample['messages']:
        role = {"system": "系统", "user": "用户", "assistant": "助手"}[msg["role"]]
        content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        print(f"  [{role}] {content}")
    print(f"\n参考回复: {sample['reference_response']}")


