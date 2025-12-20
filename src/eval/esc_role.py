"""
ESC-Role 角色扮演模块

使用ESC-ROLE 模型模拟有烦恼的用户， 与待评测的ESC模型进行多轮对话
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class ESCRole:
    """
    ESC-role 角色扮演模型
    模拟现有烦恼用户，  根据角色卡与ESC模型进行交互
    """

    def __init__(
            self,
            model_path: str,
            torch_dtype: str = "bfloat16",
            device_map: str = "auto",
            max_new_tokens: int = 512
            ):
        """初始化ESC—Role 模型

        Args:
            model_path (str): ESC-role 模型路径
            torch_dtype (str, optional): 数据类型. Defaults to "bfloat16".
            device_map (str, optional): 设备映射. Defaults to "auto".
            max_new_tokens (int, optional): 最大生成长度. Defaults to 512.
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        # 数据类型映射
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        print(f"加载 ESC-Role 模型： {model_path}")

        # 加载tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        ).eval()

        print(f"ESC-Role 模型加载完成，设备：{self.model.device}")

    def generate(
            self,
            messages: List[Dict[str, str]]
    ) -> str:
        """生成角色扮演回复

        Args:
            messages (List[Dict[str, str]]): 消息列表 [{"role": , "content": }]

        Returns:
            str: 文本回复
        """
        # 确保有 system消息
        if messages[0]["role"] != "system":
            messages.insert(0, 
                            {
                                "role": "system",
                                "content": "You are a helpful assistant!"
                            })
        # 构建输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()
    
    def unload(self):
        """卸载模型释放显存"""
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
        print("ESC-role 模型已经卸载")



class ESCModel:
    """
    待评测的ESC 模型封装
    统一接口
    包括base模型、SFT模型和RL模型
    """
    
    def __init__(
            self,
            model_path: str,
            base_model_path: Optional[str] = None,
            torch_dtype: str = "bfloat16",
            device_map: str = "auto",
            max_new_tokens: int = 512
            ):
        """初始化 ESC 

        Args:
            model_path (str): 模型路径， （LoRA权重或者完整模型）
            base_model_path (Optional[str], optional): 基座模型. Defaults to None.
            torch_dtype (str, optional): 数据类型. Defaults to "bfloat16".
            device_map (str, optional): 设备映射. Defaults to "auto".
            max_new_tokens (int, optional): 最大生成长度n_. Defaults to 512.
        """
        from peft import PeftModel

        self.max_new_tokens = max_new_tokens

        # 数据映射
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype_val = dtype_map.get(torch_dtype, torch.bfloat16)

        model_path = Path(model_path)
        is_lora = (model_path / "adapter_config.json").exists()

        if is_lora and base_model_path:
            print(f"加载LoRA 模型：{model_path}")
            print(f"基座模型：{base_model_path}")

            # 加载基座模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype_val,
                device_map=device_map,
                trust_remote_code=True
            )

            # 加载LoRA权重
            self.model = PeftModel.from_pretrained(base_model, str(model_path))

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                padding_side="left"
            )
        else:
            print(f"加载完整模型：{model_path}")

            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype_val,
                trust_remote_code=True,
                device_map=device_map
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                padding_side="left"
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print(f"ESC 模型加载完成，设备：{self.model.device}")

    def generate(
            self,
            messages: List[Dict[str, str]]
    ) -> str:
        """生成ESC回复

        Args:
            messages (List[Dict[str, str]]): 消息列表

        Returns:
            str: 生成的回复文本
        """

        # 构建输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            [text], return_tensors="pt"
        ).to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 解码
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()
    
    def unload(self):
        """卸载模型释放显存"""
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
        print("ESC 模型已经卸载")



def load_role_cards(role_cards_path: str) -> List[Dict[str, Any]]:
    """_summary_

    Args:
        role_cards_path (str): 角色卡片JSON文件地址

    Returns:
        List[Dict[str, Any]]: 角色卡列表
    """
    with open(role_cards_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"加载了 {len(data)} 个角色卡")
    return data

def run_esc_dialogue(
        esc_model: ESCModel,
        esc_role: ESCRole,
        role_card: Dict[str, Any],
        num_turns: int = 15,
        language: str = "en"
) -> List[str]:
    """运行单张角色卡的多轮对话

    Args:
        esc_model (ESCModel): 待评测的ESC模型
        esc_role (ESCRole): ESC_role 角色扮演模型
        role_card (Dict[str, Any]): 角色卡数据
        num_turns (int, optional): 对话轮次. Defaults to 15.
        language (str, optional): 语言. Defaults to "en".

    Returns:
        List[str]: 对话记录列表
    """
    # 构建 ESC-role 的 system_prompt
    if language == "zh":
        role_prompt = (
             f"我希望你扮演成一名有烦恼的人与一个AI助手交流，以下是你的角色卡：\"{role_card['base']}\"\n"
            "请尽可能模拟真实人类口语对话，不要表现得像一个机器，不要提及你是一名AI助手，"
            "在多次对话的交流中一次一次展现你的烦恼。"
        )
        initial_greeting = "你好，我是你的私人助手，您有任何烦恼都可以向我诉说！"
    else:
        role_prompt = (
            f"I hope you can role-play as a person with worries and communicate with an AI assistant. "
            f"Here is your character card: \"{role_card['base']}\"\n"
            "Please simulate a natural human conversation as much as possible. "
            "Avoid behaving like a machine and refrain from mentioning that you are an AI assistant. "
            "Gradually reveal your worries throughout our multiple conversations."
        )
        initial_greeting = "Hello, I'm your personal assistant. You can confide in me about any worries or concerns you may have!"

    # 初始化消息历史
    role_messages = [
        {
            "role": "system", "content": role_prompt
        },
        {
            "role": "user", "content": initial_greeting
        }
    ]

    esc_messages = [] # ESC 模型的消息历史
    dialogue_record = []

    # 进行多轮对话
    for turn in range(num_turns):
        # 1. ESC-role 生成用户回复
        user_response = esc_role.generate(role_messages)

        # 记录
        dialogue_record.append(f"ESC-role: {user_response}")

        # 更新 ESC-role 的历史
        role_messages.append({
            "role": "assistant",
            "content": user_response
        })

        # 更新 ESC 模型的历史 (作为用户发言)
        esc_messages.append(
            {
                "role": "user",
                "content": user_response
            }
        )

        # 2. ESC 模型生成回复
        assistant_response = esc_model.generate(esc_messages)

        # 记录
        dialogue_record.append(f"AI assistant: {assistant_response}")

        # 更新 ESC 模型的历史
        esc_messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )

        # 更新 ESC-Role 的历史（AI 助手的回复作为下一轮的 user 输入）
        role_messages.append(
            {
                "role": "user",
                "content": assistant_response
            }
        )

    return dialogue_record

def run_all_dialogues(
        esc_model: ESCModel,
        esc_role: ESCRole,
        role_cards: List[Dict[str, Any]],
        num_turns: int = 15,
        language: str = "en",
        max_samples: Optional[int] = None
        ) -> Dict[str, List[str]]:
    """对所有角色卡进行对话评测

    Args:
        esc_model (ESCModel): 待评测模型
        esc_role (ESCRole): 角色扮演模型
        role_cards (List[Dict[str, Any]]): 角色卡列表
        num_turns (int, optional): 每个对话对话轮次. Defaults to 15.
        language (str, optional): 语言. Defaults to "en".
        max_samples (Optional[int], optional): 最大样本数. Defaults to None.

    Returns:
        Dict[str, List[str]]: 对话结果字典 {index: dialogue_list}
    """
    results = {}
    
    if max_samples:
        role_cards = role_cards[: max_samples]

    for idx, role_card in enumerate(tqdm(role_cards, desc="生成对话")):
        dialogue = run_esc_dialogue(
            esc_model=esc_model,
            esc_role=esc_role,
            role_card=role_card,
            num_turns=num_turns
        )
        results[str(idx)] = dialogue
    
    return results

def save_dialogues(
     dialogues: Dict[str, List[str]],
     output_path: str   
):
    """保存对话结果"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=4)

    print(f"对话已保存到：{output_path}")
