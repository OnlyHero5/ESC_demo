"""
SFT 数据整理器

核心功能：
    1、将多轮对话转换为模型输入格式
    2、构建labels，只在supporter回复计算loss
    3、支持动态 padding 和 截断
"""

import json
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

@dataclass
class ESCSFTDataCollator:
    """
    ESC SFT 数据整理器

    将对话样本转换成模型训练格式：
        - input_ids: 完整对话 （包括历史+目标回复）
        - attention_mask: 注意力掩码
        - labels: 只在目标回复部分有效，其他位置为-100
    """
    tokenizer: PreTrainedTokenizer
    max_seq_length: int = 2048
    max_prompt_length: int = 1536
    max_response_length: int = 512
    padding: str = "max_length"

    def __post_init__(self):
        """初始化后处理"""
        # 确保tokenizer 有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def __call__(self, features: List[Dict[str, Any]])-> Dict[str, torch.tensor]:
        """处理一个Batch的样本

        Args:
            features (List[Dict[str, Any]]): 样本列表，每个样本包含messages 和 target_response

        Returns:
            Dict[str, torch.tensor]: 包括 input_ids, attention_mask, labels 字典
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            # 解析 messages
            if isinstance(feature["messages"], str):
                messages = json.loads(feature["messages"])
            else:
                messages = feature["messages"]
            
            target_response = feature["target_response"]

            # 构建完整对话（历史+目标回复）
            # full_messages = messages + [{"role": "assistant", "content": target_response}]

            # 使用 chat template 编码
            input_ids, labels = self._encode_with_labels(messages, target_response)

            # 创建 attention_mask
            attention_mask = [1]*len(input_ids)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        batch = self._pad_batch(batch_input_ids, batch_attention_mask, batch_labels)

        return batch
    
    def _encode_with_labels(
        self,
        messages: List[Dict[str, str]],
        target_response: str
    ) -> tuple:
        """
        编码对话并构建labels

        只在 target_response 部分计算 loss, 其他部分 label 设为 -100
        """

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        prompt_ids = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length
        )

        # 编码response
        response_ids = self.tokenizer.encode(
            target_response,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_response_length
        )

        # 添加 EOS token
        if response_ids[-1] != self.tokenizer.eos_token_id:
            response_ids = response_ids + [self.tokenizer.eos_token_id]
        
        # 3.拼接
        input_ids = prompt_ids + response_ids

        # 4.截断到最大长度
        if len(input_ids) > self.max_seq_length:
            # 优先保留response,截断prompt
            excess = len(input_ids) - self.max_seq_length
            prompt_ids = prompt_ids[excess:]
            input_ids = prompt_ids + response_ids
        
        # 5.构建 labels
        # prompt部分设为 -100
        # response 部分保持原始token_id
        labels = [-100]*len(prompt_ids) + response_ids

        assert len(input_ids) == len(labels), "input_ids 和 labels长度不一样"

        return input_ids, labels

    def _pad_batch(
            self,
            batch_input_ids: List[List[int]],
            batch_attention_mask: List[List[int]],
            batch_labels: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        """
        对 batch 进行 padding
        """
        # 找到最大长度
        if self.padding == "max_length":
            max_length = self.max_seq_length
        else:
            max_length = max(len(ids) for ids in batch_input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attention_mask , labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
            padding_length = max_length - len(input_ids)

            if padding_length > 0:
                # 左 padding 对于 decoder-only 模型
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
                labels = [-100] * padding_length + labels
            elif padding_length < 0:
                # 截断
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }

def create_sft_data_collator(
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        max_prompt_length: int = 1536,
        max_response_lenth: int = 512
    ) -> ESCSFTDataCollator:
    """创建 SFT 数据整理器的工厂函数"""
    return ESCSFTDataCollator(
        tokenizer = tokenizer,
        max_seq_length = max_seq_length,
        max_prompt_length = max_prompt_length,
        max_response_length = max_response_lenth
    )