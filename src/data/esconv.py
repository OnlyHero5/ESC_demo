"""
ESConv 数据集处理模块
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, DatasetDict
from tqdm import tqdm


STARTEGY_MAP = {
    "Question": "提问",
    "Retatement or Paraphrasing": "释义",
    "Reflection of feelings": "情感反映",
    "Self-disclosure": "自我表露",
    "Affirmation and Reassurance": "肯定与安慰",
    "Providing Suggestions": "提供建议",
    "Information": "提供信息",
    "Others": "其他",
}

def load_esconv_raw(data_path: str) -> List[Dict]:
    """加载ESConv原始数据集

    Args:
        data_path (str): json路径

    Returns:
        List[Dict]: 原始对话列表
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("="*60)
    print(f"加载了 {len(data)} 个对话")
    print("="*60)

    return data



def parse_esconv_dialog(dialog: Dict, index: int) -> Dict[str, Any]:
    """解析单个对话

    Args:
        dialog (Dict): 原始对话数据
        index (int): 用于生成id

    Returns:
        Dict[str, Any]: 标准化后的对话数据
    """

    # 使用索引作为对话ID
    dialog_id = f"esconv_{index:04d}"

    # 提取情境信息
    situation = dialog.get("situation", "").strip()
    emotion_type = dialog.get("emotion_type", "")
    problem_type = dialog.get("problem_type", "")
    experience_type = dialog.get("experience_type", "")

    # 解析对话轮次
    turns = []
    raw_dialog = dialog.get("dialog", [])

    for turn in  raw_dialog:
        # 获取说话者
        speaker = turn.get("speaker","").lower().strip()

        # 获取文本内容
        text = turn.get("content", "").strip()

        # 获取策略
        annotation = turn.get("annotation", {})
        strategy = ""
        if isinstance(annotation, dict):
            strategy = annotation.get("strategy")
        
        # 应对可能的空轮次
        if not text:
            continue

        # 标准化角色名称
        if speaker in ["seeker", "usr", "user", "help-seeker"]:
            speaker = "seeker"
        elif speaker in ["supporter", "sys", "system", "helper"]:
            speaker = "supporter"
        else:
            # 跳过未知角色
            continue

        turns.append({
            "speaker": speaker,
            "text": text,
            "strategy": strategy if speaker == "supporter" else ""
         })
        
    return {
        "dialog_id": dialog_id,
        "situation": situation,
        "emotion_type": emotion_type,
        "problem_type": problem_type,
        "experience_type": experience_type,
        "turns": turns,
        "num_turns": len(turns)
    }



def convert_to_chat_format(
        dialog: Dict[str, Any],
        system_prompt: str,
        include_situation: bool = True
) -> List[Dict[str, Any]]:
    """将对话转换成多个训练样本

    每个样本包含：
    - 到当前轮为止的历史对话
    - 当前supporter的回复作为目标

    Args:
        dialog (Dict[str, Any]): 解析后的对话数据
        system_prompt (str): 系统提示词
        include_situation (bool, optional): 是否在系统提示词中包含情境描述. Defaults to True.

    Returns:
        List[Dict[str, Any]]: 训练样本
    """
    samples = []
    turns = dialog["turns"]

    #构建系统提示
    if include_situation and dialog["situation"]:
        full_system_prompt = f"{system_prompt} \n\n 【用户情境】{dialog['situation']}"
    else:
        full_system_prompt = system_prompt

    # 遍历对话，为每个 supporter回复创建训练样本
    history = []   

    for i, turn in enumerate(turns):
        if turn["speaker"] == "supporter":
            # 只有当历史中有seeker消息时才创建样本
            has_seeker_history = any(h["speaker"] == "seeker" for h in history)

            if has_seeker_history or len(history) == 0:
                #创建训练样本
                messages = [
                    {
                        "role": "system",
                        "content": full_system_prompt
                    }
                ]

                # 添加历史消息
                for h in history:
                    role = "user" if h["speaker"] == "seeker" else "assistant"
                    messages.append(
                        {
                            "role": role,
                            "content": h["text"]
                        }
                    )
                
                # 目标回复
                target_response = turn["text"]
                strategy = turn["strategy"]

                samples.append(
                    {
                        "dialog_id": dialog["dialog_id"],
                        "turn_index": i,
                        "messages": messages,
                        "target_response": target_response,
                        "strategy": strategy,
                        "emotion_type": dialog["emotion_type"],
                        "problem_type": dialog["problem_type"]
                    }
                )
        history.append(turn)
    
    return samples



def build_sft_dataset(
        dialogs: List[Dict[str, Any]],
        system_prompt: str,
        include_situation: bool = True,
) -> List[Dict[str, Any]]:
    """构建SFT训练数据集

    Args:
        dialogs (List[Dict[str, Any]]): _解析后的对话列表
        system_prompt (str): 系统提示词
        include_situation (bool, optional): 是否包含情境描述. Defaults to True.

    Returns:
        List[Dict[str, Any]]: SFT 训练样本列表
    """

    all_samples = []

    for dialog in tqdm(dialogs, desc="构建SFT数据集"):
        samples = convert_to_chat_format(
            dialog,
            system_prompt,
            include_situation
        )
        all_samples.extend(samples)
    
    print("="*60)
    print(f" 共生成 {len(all_samples)} 个 SFT 训练样本")
    print("="*60)
    return all_samples



def split_dataset(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按对话维度划分数据集

    Args:
        samples (List[Dict[str, Any]]): 所有样本
        train_ratio (float, optional): 训练集比例. Defaults to 0.8.
        val_ratio (float, optional): 验证集比例. Defaults to 0.1.
        test_ratio (float, optional): 测试集比例. Defaults to 0.1.
        seed (int, optional): 随机种子. Defaults to 42.

    Returns:
        Tuple[List[Dict], List[Dict], List[Dict]]: (train_samples, val_sample, test_samples)
    """
    import random
    random.seed(seed)

    #按对话ID划分
    dialog_samples = {}
    for sample in samples:
        dialog_id = sample["dialog_id"]

        if dialog_id not in dialog_samples:
            dialog_samples[dialog_id] = []

        dialog_samples[dialog_id].append(sample)

    # 打乱顺序
    dialog_ids = list(dialog_samples.keys())
    random.shuffle(dialog_ids)

    # 计算划分点
    n_dialogs = len(dialog_ids)
    n_train = int(n_dialogs * train_ratio)
    n_val = int(n_dialogs * val_ratio)

    train_ids = dialog_ids[:n_train]
    val_ids = dialog_ids[n_train:n_train+n_val]
    test_ids = dialog_ids[n_train+n_val:]

    # 收集样本
    train_samples = [s for did in train_ids for s in dialog_samples[did]]
    val_samples = [s for did in val_ids for s in dialog_samples[did]]
    test_samples = [s for did in test_ids for s in dialog_samples[did]]

    print("="*60)
    print(f" 数据集划分：")
    print(f" 训练集：{len(train_samples)}样本 ({len(train_ids)} 对话)")
    print(f" 验证集：{len(val_samples)} 样本 ({len(val_ids)} 对话)")
    print(f" 测试集: {len(test_samples)} 样本 ({len(test_ids)} 对话)")

    return train_samples, val_samples, test_samples



def create_hf_dataset(
        samples: List[Dict[str, Any]]
) -> Dataset:
    """列表转huggingface dataset

    Args:
        samples (List[Dict[str, Any]]): 样本列表

    Returns:
        Dataset: huggingface Dataset
    """
    processed_samples = []
    for sample in samples:
        processed_samples.append({
            "dialog_id": sample["dialog_id"],
            "turn_index": sample["turn_index"],
            "messages": json.dumps(sample["messages"], ensure_ascii=False),
            "target_response": sample["target_response"],
            "strategy": sample["strategy"],
            "emotion_type": sample.get("emotion_type", ""),
            "problem_type": sample.get("problem_type", ""),
        })
    return Dataset.from_list(processed_samples)



def load_esconv(
        data_path: str = "data/esconv/raw/ESConv.json",
        system_prompt: Optional[str] = None,
        include_situation: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        save_processed: bool = True,
        processed_dir: str = "data/esconv/processed",
) -> DatasetDict:
    """加载并处理ESConv数据集

    Args:
        data_path (str, optional): 原始数据文件路径. Defaults to "data/esconv/raw/ESConv.json".
        system_prompt (Optional[str], optional): 系统提示词. Defaults to None.
        include_situation (bool, optional): 系统提示里是否包含情境描述. Defaults to True.
        train_ratio (float, optional): 训练集比例. Defaults to 0.8.
        val_ratio (float, optional): 验证集比例. Defaults to 0.1.
        test_ratio (float, optional): 测试集比例. Defaults to 0.1.
        seed (int, optional): 随机种子. Defaults to 42.
        save_processed (bool, optional): 是否保存处理后的数据. Defaults to True.
        processed_dir (str, optional): 处理后数据保存目录. Defaults to "data/esconv/processed".

    Returns:
        DatasetDict
    """
    # 默认系统提示词
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
        print("ESConv 数据集")
        print("="*60)

        # 1.加载原始数据集
        raw_data = load_esconv_raw(data_path)

        # 2.解析对话（使用索引作为dialog_id）
        print("\n 解析对话...")
        parsed_dialogs = []
        for idx, d in enumerate(tqdm(raw_data, desc="解析对话")):
            parsed = parse_esconv_dialog(d, idx)
            parsed_dialogs.append(parsed)
        
        # 过滤无效对话
        valid_dialogs = [d for d in parsed_dialogs if d["num_turns"] >= 2]
        print(f" 有效对话：{len(valid_dialogs)} / {len(parsed_dialogs)}")

        # 3. 构建SFT数据集
        print("\n 构建SFT样本...")
        all_samples = build_sft_dataset(
            valid_dialogs,
            system_prompt,
            include_situation
        )

        # 4.划分数据集
        print("\n 划分数据集")
        train_samples, val_samples, test_samples = split_dataset(
            all_samples,
            train_ratio,
            val_ratio,
            test_ratio,
            seed
        )

        # 5. 创建DatasetDict
        dataset_dict = DatasetDict(
            {
            "train": create_hf_dataset(train_samples),
            "validation": create_hf_dataset(val_samples),
            "test": create_hf_dataset(test_samples)
            }
        )

        # 6. 保存处理后的数据
        if save_processed:
            processed_path = Path(processed_dir)
            processed_path.mkdir(parents=True, exist_ok=True)
            dataset_dict.save_to_disk(str(processed_path))
            print(f"\n 数据已保存到：{processed_path}")
        
        print("\n" + "="*60)
        print(" ESConv 数据集处理完成！")
        print("="*60)

        return dataset_dict
    


def load_processed_esconv(
        processed_dir: str="data/esconv/processed"
) -> DatasetDict:
    """加载已构建好的ESConv数据集

    Args:
        processed_dir (str, optional): 数据集目录. Defaults to "data/esconv/processed".

    Returns:
        DatasetDict
    """
    from datasets import load_from_disk

    processed_path = Path(processed_dir)
    if not processed_path.exists():
        raise FileNotFoundError(f"处理后的数据不存在：{processed_dir}")
    
    dataset = load_from_disk(str(processed_dir))
    print(" 已加载处理后的ESConv数据集")
    print(f" 训练集：{len(dataset['train'])} 样本")
    print(f" 验证集：{len(dataset['validation'])} 样本")
    print(f" 测试集：{len(dataset['test'])} 样本")

    return dataset



def get_sample_dialog(
        dataset: DatasetDict,
        split: str = "train",
        index: int = 0
        ) -> Dict:
    """捕获一个样本用于测试

    Args:
        dataset (DatasetDict): 
        split (str, optional):  Defaults to "train".
        index (int, optional):  Defaults to 0.

    Returns:
        Dict: 
    """
    sample = dict(dataset[split][index])
    sample["messages"] = json.loads(sample["messages"])
    return sample



#========================统计分析函数============
def analyze_esconv_dataset(dataset: Dataset) -> Dict[str, Any]:
    """数据统计

    Args:
        dataset (Dataset):
    Returns:
        Dict[str, Any]: 
    """
    stats = {
        "splits": {},
        "strategies": {},
        "emotion_types": {},
        "problem_types": {},
        "turn_lengths": [],
        "response_lengths": []
    }

    for split_name in dataset.keys():
        split_data = dataset[split_name]
        stats["splits"][split_name] = len(split_data)

        for sample in split_data:
            # 统计策略分布
            strategy = sample["strategy"]
            if strategy:
                stats["strategies"][strategy] = stats["strategies"].get(strategy, 0) + 1
            # 统计情感类型
            emotion = sample.get("emotion_type", "")
            if emotion:
                stats["emotion_types"][emotion] = stats["emotion_types"].get(emotion, 0) + 1
            # 统计问题类型
            problem = sample.get("problem_type", "")
            if emotion:
                stats["problem_types"][problem] = stats["problem_types"].get(problem, 0) + 1

            # 统计对话轮次
            messages = json.loads(sample["messages"])
            stats["turn_lengths"].append(len(messages))

            # 统计回复长度
            stats["response_lengths"].append(len(sample["target_response"]))
            
    # 计算平均值
    if stats["turn_lengths"]:
        stats["avg_turn_length"] = sum(stats["turn_lengths"]) / len(stats["turn_lengths"])
    if stats["response_lengths"]:
        stats["avg_response_length"] = sum(stats["response_lengths"]) / len(stats["response_lengths"])
    
    return stats



def print_dataset_stats(stats: Dict[str, Any]):
    """打印数据集统计信息"""
    print("\n" + "="*60)
    print("ESConv 数据集统计")
    print("="*60)

    print("\n 【数据划分】")
    for split, count in stats["splits"].items():
        print(f"    {split}: {count} 样本")
    
    print("\n 【平均统计】")
    print(f" 平均上下文轮次：{stats.get('avg_turn_length', 0):.1f}")
    print(f" 平均回复长度： {stats.get('avg_response_length', 0):.1f} 字符")

    print("\n 【策略分布】")
    sorted_strategies = sorted(stats["strategies"].items(), key=lambda x: x[1], reverse=True)
    total = sum(stats["strategies"].values())
    for strategy, count in sorted_strategies[:10]:
        pct = 100 * count / total
        strategy_cn = STARTEGY_MAP.get(strategy, strategy)
        print(f" {strategy_cn}: {count} {pct:.1f}%")
    
    print("\n 【情感类型分布】")
    sorted_emotions = sorted(stats["emotion_types"].items(), key=lambda x: x[1], reverse=True)
    for emotion, count in sorted_emotions[:len(sorted_emotions)]:
        print(f" {emotion}: {count}")

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    #处理数据集
    dataset = load_esconv(
        data_path="data/esconv/raw/ESConv.json",
        save_processed=True
    )

    # 打印统计信息
    stats = analyze_esconv_dataset(dataset)
    print_dataset_stats(stats)

    print("\n" + "=" * 60)
    print("📝 样本示例")
    print("=" * 60)
    sample = get_sample_dialog(dataset, "train", 0)
    print(f"\n对话 ID: {sample['dialog_id']}")
    print(f"轮次索引: {sample['turn_index']}")
    print(f"策略: {sample['strategy']}")
    print(f"情感类型: {sample['emotion_type']}")
    print(f"\n历史消息:")
    for msg in sample['messages']:
        role = {"system": "系统", "user": "用户", "assistant": "助手"}[msg["role"]]
        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(f"  [{role}] {content}")
    print(f"\n目标回复: {sample['target_response']}")