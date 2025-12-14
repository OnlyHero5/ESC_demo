"""
ESC情感支持对话奖励函数

奖励组成：
1. 参考答案匹配
2. 多样性奖励
3. 长度惩罚
4. 连贯性奖励
"""

import re
import threading
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

class ESCRewardFunction:
    """
    ESC 任务奖励函数

    奖励组成：
        1. 参考答案匹配
        2. 多样性奖励
        3. 长度惩罚
        4. 连贯性奖励
    """
    
    def __init__(
            self,
            reward_weights: Dict[str, float] = None,
            length_config: Dict[str, int] = None,
            negative_keywords: List[str] = None,
            scale_rewards: bool = True,
            reward_clip: float = 10.0
    ):
        """初始化奖励函数

        Args:
            reward_weights (Dict[str, float], optional): 各奖励成分权重. Defaults to None.
            length_config (Dict[str, int], optional): 长度配置. Defaults to None.
            negative_keywords (List[str], optional): 负面关键词列表. Defaults to None.
            scale_rewards (bool, optional): 是否缩放奖励. Defaults to True.
            reward_clip (float, optional): 奖励裁剪范围. Defaults to 10.0.
        """
        # 默认权重
        self.reward_weights = reward_weights or {
            "bleu": 0.3,
            "rouge": 0.3,
            "distinct": 0.2,
            "length_penalty": 0.1,
            "coherence": 0.1
        }
        # 长度配置
        self.length_config = length_config or {
            "min_length": 10,
            "max_length": 300,
            "optimal_length": 80
        }
        # 负面关键词
        self.negative_keywords = negative_keywords or [
    "i don't know", "i'm not sure", "i can't help",
    "no idea", "not sure about that", "don't have that information",
    "unable to help", "can't assist with that", "can't provide that",
    "sorry", "apologies", "i cannot", "i can't do that"
]


        self.scale_rewards = scale_rewards

        self.reward_clip = reward_clip

        self.bleu_metric = BLEU(effective_order=True)

        self._TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")

        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    
    def compute_bleu_reward(
            self,
            hypothesis: str,
            reference: str
    ) -> float:
        if not hypothesis.strip() or not reference.strip():
           return 0.0
        
        return self.bleu_metric.sentence_score(hypothesis, [reference]).score
    
    def compute_rouge_reward(
            self,
            hypothesis: str,
            reference: str
    ) -> float:
        """计算ROUGE-L奖励"""
        if not hypothesis.strip() or not reference.strip():
            return 0.0
        
        score = self.scorer.score(reference, hypothesis)["rougeL"]
        return score.fmeasure * 100
    
    def compute_distinct_reward(
            self,
            hypothesis: str
    ) -> float:
        """计算 Distinct 奖励"""
        words = hypothesis.lower().split()

        if len(words) < 2:
            return 0.0
        
        # Distinct-1
        unigrams = words
        distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0

        # Distinct-2
        bigrams = [tuple(words[i:i+2]) for i in range(len(words) -1)]
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0

        # 平均
        return (distinct_1 + distinct_2) / 2 * 100
    
    def compute_length_penalty(
            self,
            hypothesis: str
    ) -> float:
        """计算长度惩罚"""
        length = len(hypothesis)
        min_len = self.length_config["min_length"]
        max_len = self.length_config["max_length"]
        optimal_len = self.length_config["optimal_length"]

        if length < min_len:
            return -50.0 * (1 - length/ min_len)
        elif length > max_len:
            return -20.0 * (length - max_len) / max_len
        else:
            distance = abs(length - optimal_len) / optimal_len
            return 10*(1-distance)
    
    def _tokens(self, text: str) -> list[str]:
    # 兼容中英文：中文按连续汉字片段，英文按单词/数字
        return self._TOKEN_RE.findall(text.lower())

    def compute_coherence_reward(self, hypothesis: str, context: Optional[str] = None) -> float:
        """计算连贯性奖励 和 上下文相关性奖励"""
        reward = 50.0

        hyp_text = hypothesis.strip()
        hyp_tokens = self._tokens(hyp_text)
        hyp_set = set(hyp_tokens)

        # 1) 负面关键词（建议：若 negative_keywords 是英文词，可考虑改成 token/词边界匹配）
        hyp_lower = hypothesis.lower()
        for keyword in self.negative_keywords:
            if keyword.lower() in hyp_lower:
                reward -= 20.0

        # 2) 检查连续三连词（边界修正：>=3）
        if len(hyp_tokens) >= 3:
            for i in range(len(hyp_tokens) - 2):
                if hyp_tokens[i] == hyp_tokens[i + 1] == hyp_tokens[i + 2]:
                    reward -= 30.0
                    break

        # 3) 是否为空或太简短
        if len(hyp_text) < 5:
            reward -= 50.0

        # 4) 上下文相关性
        if context:
            ctx_tokens = self._tokens(context)
            ctx_set = set(ctx_tokens)

            # 默认值，避免 UnboundLocalError :contentReference[oaicite:4]{index=4}
            overlap_ratio = 0.0
            if hyp_set:
                overlap_ratio = len(ctx_set & hyp_set) / len(hyp_set)  # set 交集 :contentReference[oaicite:5]{index=5}

            # 4.1 简单复制上下文（对很短回答别误伤）
            if len(hyp_set) >= 6 and overlap_ratio > 0.8:
                reward -= 25.0

            # 4.2 完全无关（同样加长度门槛）
            if len(hyp_set) >= 6 and len(ctx_set) >= 6 and overlap_ratio < 0.05:
                reward -= 15.0

            # 4.3 检查是否直接重复用户最后一句话（取最后一次 marker）
            context_lower = context.lower()
            user_markers = ["user:", "[用户]", "seeker:"]
            last_pos, last_marker = -1, None
            for m in user_markers:
                p = context_lower.rfind(m)
                if p > last_pos:
                    last_pos, last_marker = p, m

            last_user_utterance = ""
            if last_marker is not None and last_pos >= 0:
                after = context_lower[last_pos + len(last_marker):]
                last_user_utterance = after.split("\n", 1)[0].strip()

            if last_user_utterance:
                user_tokens = set(self._tokens(last_user_utterance))
                if user_tokens:
                    user_overlap = len(user_tokens & hyp_set) / len(user_tokens)
                    if len(user_tokens) >= 6 and user_overlap > 0.7:
                        reward -= 20.0

            # 奖励适度相关（确保 overlap_ratio 一定有值）
            if 0.1 <= overlap_ratio <= 0.5:
                reward += 10.0

            # “重要词”示例：这里仍是简单启发式（更建议做停用词过滤/TF-IDF/实体识别）
            important = [w for w in ctx_set if len(w) > 4]
            mentioned = sum(1 for w in important if w in hyp_set)
            if mentioned > 0:
                reward += min(10.0, mentioned * 3.0)

        return max(reward, 0.0)

    def compute_reward(
            self,
            hypothesis: str,
            reference: str,
            context: Optional[str] = None,
    ) -> Dict[str, float]:
        """计算综合奖励分数

        Args:
            hypothesis (str): 模型生成的回复
            reference (str): 参考回复
            context (Optional[str], optional): 对话上下文. Defaults to None.

        Returns:
            Dict[str, float]: 各成分奖励和总奖励的字典
        """
        rewards = {}

        # 计算各成分奖励
        rewards["bleu"] = self.compute_bleu_reward(hypothesis, reference)
        rewards["rouge"] = self.compute_rouge_reward(hypothesis, reference)
        rewards["distinct"] = self.compute_distinct_reward(hypothesis)
        rewards["length_penalty"] = self.compute_length_penalty(hypothesis)
        rewards["coherence"] = self.compute_coherence_reward(hypothesis, context)

        # 加权求和
        total = sum(
            self.reward_weights.get(key, 0) * value
            for key, value in rewards.items()
        )

        # 缩放和裁剪
        if self.scale_rewards:
            total = total / 100.0
        if self.reward_clip:
            total = max(-self.reward_clip, min(self.reward_clip, total))
        
        rewards["total"] = total

        return rewards
    
    def __call__(
            self,
            completions: List[str] = None,
            prompts: Optional[List[str]] = None,
            references: Optional[List[str]] = None,
            ** kwargs
    ) -> List[float]:
        """批量计算奖励

        Args:
            completions (List[str], optional): 生成的回复列表. Defaults to None.
            prompts (Optional[List[str]], optional): 提示词列表. Defaults to None.
            references (Optional[List[str]], optional): 参考回复列表. Defaults to None.

        Returns:
            List[float]: 奖励分数列表_
        """
        rewards = []

        for i, completion in enumerate(completions):
            # 获取参考回复
            ref = references[i] if references and i < len(references) else ""
            context = prompts[i] if prompts and i < len(prompts) else None

            # 计算奖励
            reward_dict = self.compute_reward(completion, ref, context)
            rewards.append(reward_dict["total"])
        
        return rewards
    


# 全局单例容器
class ScorerContainer:
    _instance = None
    _lock = threading.Lock()
def get_scorer_instance(config: Dict[str, Any] = None):
    """单例获取器：确保ESCRewardFunction只会被初始化一次"""
    if ScorerContainer._instance is None:
        if config is None:
            config = {}
        
        grpo_config = config.get("grpo", {})

        # 初始化耗时的类
        ScorerContainer._instance = ESCRewardFunction(
            reward_weights=grpo_config.get("reward_weights", {}),
            scale_rewards=grpo_config.get("scale_rewards", True),
            reward_clip=grpo_config.get("reward_clip", 10.0)
        )

        print("="*60)
        print("ESCRewardFunction 已经单例初始化完成")
        print("="*60)
    
    return ScorerContainer._instance

def _apply_clip(value: float, scorer) -> float:
    """应用奖励裁剪"""
    if scorer.reward_clip:
        return max(-scorer.reward_clip, min(scorer.reward_clip, value))
    return value

def create_reward_functions(config: Dict[str, Any]) -> List[Any]:
    
    get_scorer_instance(config)

    return [
        bleu_reward_func,
        rouge_reward_func,
        distinct_reward_func,
        length_reward_func,
        coherence_reward_func
    ]

# =========================================================================
# 独立奖励函数 (Adapters for TRL)
# 规范：
# 1. 必须接受 completions, **kwargs
# 2. 根据需要接受 prompts (即 context) 或 references
# 3. 必须返回 List[float]
# 4. 内部负责调用单例并应用权重
# =========================================================================
def bleu_reward_func(completions: List[str], references: List[str], **kwargs) -> List[float]:
    scorer = get_scorer_instance()
    weight = scorer.reward_weights.get("bleu", 0.0)
    rewards = []

    refs = references if references else [""]*len(completions)

    for com, ref in zip(completions, refs):
        raw_score = scorer.compute_bleu_reward(com, ref)
        # 应用缩放
        if scorer.scale_rewards:
            r = (raw_score / 100.0) * weight
        else:
            r = raw_score * weight
        # 应用裁剪
        r = _apply_clip(r, scorer)
        rewards.append(r)
    
    return rewards

def rouge_reward_func(completions: List[str], references: List[str], **kwargs) -> List[float]:
    scorer = get_scorer_instance()
    weight = scorer.reward_weights.get("rouge", 0.0)
    rewards = []

    refs = references if references else [""]*len(completions)

    for com, ref in zip(completions, refs):
        raw_score = scorer.compute_rouge_reward(com, ref)
        # 应用缩放
        if scorer.scale_rewards:
            r = (raw_score / 100.0) * weight
        else:
            r = raw_score * weight
        # 应用裁剪
        r = _apply_clip(r, scorer)
        rewards.append(r)
    
    return rewards

def distinct_reward_func(completions: List[str], **kwargs) -> List[float]:
    scorer = get_scorer_instance()
    weight = scorer.reward_weights.get("distinct", 0.0)
    rewards = []

    for com in completions:
        raw_score = scorer.compute_distinct_reward(com)
        # 应用缩放
        if scorer.scale_rewards:
            r = (raw_score / 100.0) * weight
        else:
            r = raw_score * weight
        # 应用裁剪
        r = _apply_clip(r, scorer)
        rewards.append(r)
    
    
    return rewards
    
def length_reward_func(completions: List[str], **kwargs) -> List[float]:
    scorer = get_scorer_instance()
    weight = scorer.reward_weights.get("length_penalty", 0.0)
    rewards = []

    for com in completions:
        raw_score = scorer.compute_length_penalty(com)
        # 应用缩放（length_penalty 范围约 -50 ~ +10，用 50 归一化）
        if scorer.scale_rewards:
            r = (raw_score / 50.0) * weight
        else:
            r = raw_score * weight
        # 应用裁剪
        r = _apply_clip(r, scorer)
        rewards.append(r)
    
    return rewards
    
def coherence_reward_func(completions: List[str], prompts: List[str] = None,**kwargs) -> List[float]:

    scorer = get_scorer_instance()
    weight = scorer.reward_weights.get("coherence", 0.0)
    rewards = []

    ctxs = prompts if prompts else [None]*len(completions)
    for com, ctx in zip(completions, ctxs):
        raw_score = scorer.compute_coherence_reward(com, ctx)
        # 应用缩放（coherence 范围约 0 ~ 70+，用 50 归一化）
        if scorer.scale_rewards:
            r = (raw_score / 50.0) * weight
        else:
            r = raw_score * weight
        # 应用裁剪
        r = _apply_clip(r, scorer)
        rewards.append(r)

    return rewards

