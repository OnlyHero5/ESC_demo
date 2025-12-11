"""
ESC评测指标

包含:
    - BLEU-2, BLEU-4,
    - ROUGE-L
    - Distinct-1, Distinct-2
    - BertScore
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np

def compute_bleu(
        references: List[str],
        hypotheses: List[str],
        max_n: int = 4
) -> Dict[str, float]:
    """计算BLEU分数

    Args:
        references (List[str]): 参考回复列表
        hypotheses (List[str]): 生成回复列表
        max_n (int, optional): 最大n-gram. Defaults to 4.

    Returns:
        Dict[str, float]: 包含BLEU-1 到 BLEU-n的字典
    """
    try:
        from sacrebleu.metrics import BLEU

        refs = [[ref] for ref in references]

        bleu = BLEU(effective_order=True)
        result = bleu.corpus_score(hypotheses, [[r[0] for r in refs]])

        scores = {
            "bleu": result.score,
            "bleu_1": result.precisions[0] if len(result.precisions) > 0 else 0,
            "bleu_2": result.precisions[1] if len(result.precisions) > 1 else 0,
            "bleu_3": result.precisions[2] if len(result.precisions) > 2 else 0,
            "bleu_4": result.precisions[3] if len(result.precisions) > 3 else 0,
        }
        
        return scores
    
    except ImportError:
        print("警告: sacrebleu 未安装")
        return {"bleu": 0, "bleu_1": 0, "bleu_2": 0, "bleu_3": 0, "bleu_4": 0}
    
def compute_rouge(
        references: List[str],
        hypotheses: List[str]
) -> Dict[str, float]:
    """计算 ROUGE 分数

    Args:
        references (List[str]): 参考回复列表
        hypotheses (List[str]): 生成回复列表

    Returns:
        Dict[str, float]: 包含 ROUGE-1, ROUGE-2, ROUGE-L 字典
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

        scores = {
            "rouge_1": [],
            "rouge_2": [],
            "rouge_l": []
        }

        for ref, hyp in zip(references, hypotheses):
            result = scorer.score(ref, hyp)
            scores["rouge_1"].append(result["rouge1"].fmeasure)
            scores["rouge_2"].append(result["rouge2"].fmeasure)
            scores["rouge_l"].append(result["rougeL"].fmeasure)
        
        return {
            "rouge_1": np.mean(scores["rouge_1"]) * 100,
            "rouge_2": np.mean(scores["rouge_2"]) * 100,
            "rouge_l": np.mean(scores["rouge_l"]) * 100,
        }
    
    except ImportError:
        print("警告： rouge-score 未安装")
        return {
            "rouge_1": 0,
            "rouge_2": 0,
            "rouge_l": 0
        }

def compute_distinct(
        hypotheses: List[str],
        max_n: int = 2
) -> Dict[str, float]:
    """计算 Distinct-n分数

    Args:
        hypotheses (Listp[str]): 生成回复列表
        max_n (int, optional): 最大n-gram. Defaults to 2.

    Returns:
        Dict[str, float]: 包含Distinct-1, Distinct-2等

    """
    def get_ngrams(text: str, n: int) -> List[tuple]:
        words = text.lower().split()
        return [tuple(words[i:i+n]) for i in range(len(words)- n + 1)]
    
    scores = {}

    for n in range(1, max_n + 1):
        all_ngrams = []
        for hyp in hypotheses:
            all_ngrams.extend(get_ngrams(hyp, n))
        
        if len(all_ngrams) > 0:
            distinct = len(set(all_ngrams)) / len(all_ngrams)
        else:
            distinct = 0
        
        scores[f"distinct_{n}"] = distinct * 100
    
    return scores

def compute_bertscore(
        references: List[str],
        hypotheses: List[str],
        lang: str = "en",
        device: str = "cuda"
) -> Dict[str, float]:
    """计算BERTScore

    Args:
        references (List[str]): 参考回复列表
        hypotheses (List[str]): 生成回复列表
        lang (str, optional): 语言. Defaults to "en".
        device (str, optional): 计算设备. Defaults to "cuda".

    Returns:
        Dict[str, float]: BERTScore, P/R/F1 的字典
    """
    try:
        from bert_score import score

        P, R, F1 = score(
            hypotheses,
            references,
            lang=lang,
            device=device,
            verbose=False
        )

        return {
            "bertscore_p": P.mean().item() * 100,
            "bertscore_r": R.mean().item() * 100,
            "bertscore_f1": F1.mean().item() * 100
        }
    
    except ImportError:
        print("警告：bert-score 未安装")
        return {
            "bertscore_p": 0,
            "bertscore_r": 0,
            "bertscore_f1": 0
        }
    except Exception as e:
        print(f"BERTScore 计算失败：{e}")
        return {
            "bertscore_p": P.mean().item() * 100,
            "bertscore_r": R.mean().item() * 100,
            "bertscore_f1": F1.mean().item() * 100
        }

def compute_all_metrics(
        references: List[str],
        hypotheses: List[str],
        include_bertscore: bool = True,
        lang: str = "en"
) -> Dict[str, float]:
    """计算所有评测指标

    Args:
        references (List[str]): 参考回复列表
        hypotheses (List[str]): 生成回复列表
        include_bertscore (bool, optional): 是否包含bertscore. Defaults to True.
        lang (str, optional): 语言. Defaults to "en".

    Returns:
        Dict[str, float]: 所有指标的字典
    """
    metrics = {}

    # BLEU
    bleu_scores = compute_bleu(references, hypotheses)
    metrics.update(bleu_scores)

    # ROUGE
    rouge_scores = compute_rouge(references, hypotheses)
    metrics.update(rouge_scores)

    # Distinct
    distinct_scores = compute_distinct(hypotheses)

    # BERTScore
    if include_bertscore:
        bert_scores = compute_bertscore(references, hypotheses, lang=lang)
        metrics.update(bert_scores)
    
    return metrics

def print_metrics(
        metrics: Dict[str, float],
        title: str = "评测结果"
):
    """打印评测结果"""
    print("\n" + "=" * 60)
    print(f"{title}")
    print("="*60)

    # BLEU
    print("\n【BLEU】")
    for key in ["bleu", "bleu_2", "bleu_4"]:
        if key in metrics:
            print(f"{key.upper()} : {metrics[key]:.2f}")
    
    # Distinct
    print("\n 【Distinct】")
    for key in ["distinct_1", "distinct_2"]:
        if key in metrics:
            print(f"{key.upper()}: {metrics[key]:.2f}")
    
    # BERTScore
    if "bertscore_f1" in metrics:
        print("\n 【BERTScore】")
        print(f" F1: {metrics['bertscore_f1']:.2f}")
    
    print("="*60)
