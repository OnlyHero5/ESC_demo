"""
Qwen3封装类, 支持Chat模式和训练模式
"""


import torch
from typing import List, Dict, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

class QwenChatModel:
    """
    Qwen3 对话模型封装

    支持功能：
    - 加载预训练模型
    - Chat模式推理
    - 训练模式(返回logits)
    - LoRA 微调
    """
    
    def __init__(self,
                 model_path: str,
                 device_map: str = "auto",
                 torch_dtype: str = "bfloat16",
                 load_in_4bit: bool = False,
                 load_in_8bit: bool = False,
                 use_flash_attention: bool = True,):
        """
        初始化模型

        Args:
            model_path (str): 模型路径
            device_map (str, optional): 设备映射. Defaults to "auto".
            torch_dtype (str, optional): 数据类型. Defaults to "bfloat16" choices: ["float16","float32"].
            load_in_4bit (bool, optional): 4bit量化. Defaults to False.
            load_in_8bit (bool, optional): 8bit量化. Defaults to False.
            use_flash_attention (bool, optional): flash attention加速. Defaults to True.
        """
        self.model_path = model_path
        self.device_map = device_map

        # 设置数据类型
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 配置量化
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": self.torch_dtype,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )

        self._is_peft_model = False

        print("="*50)
        print(f"    模型加载完成：{model_path}")
        print(f"    设备：{self.model.device}")
        print(f"    数据类型：{self.torch_dtype}")
        print(f"    参数量：{self.model.num_parameters() / 1e9:.2f}")
        print("="*50)



    def apply_lora(
            self, 
            r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05,
            target_modules: Optional[List[str]] = None,
            ):
        """
        应用LoRA微调

        Args:
            r (int, optional): _description_. Defaults to 16.
            lora_alpha (int, optional): _description_. Defaults to 32.
            lora_dropout (float, optional): _description_. Defaults to 0.05.
            target_modules (Optional[List[str]], optional): 目标模块列表. Defaults to None.
        """
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self._is_peft_model = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print("="*50)
        print(f" Lora 已应用")
        print(f" 可训练参数: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")


    

    def load_lora_weights(
            self,
            lora_path: str,
    ):
        """
        加载已训练好的LoRA权重

        Args:
            lora_path (str): LoRA权重路径
        """
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_path
        )
        self._is_peft_model = True
        print("="*50)
        print(f" Lora 权重加载完成：{lora_path}")
        print("="*50)



    def build_chat_input(
            self,
            messages: List[Dict[str, str]],
            enable_thinking: bool= False,
    ) -> Dict[str, torch.Tensor]:
        """
        构建chat格式输入

        Args:
            messages (List[Dict[str, str]]): 消息列表,格式 [{"role": "user/assistant/system", "content": "..."}]
            enable_thinking (bool, optional): 是否启用思考模式. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: 包含 input_ids, attention_mask 字典
        """
        encoded = self.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",      # 让模板直接返回 torch.Tensor
        padding=True,
        truncation=True,
        max_length=2048,
        )

        if isinstance(encoded, torch.Tensor):
            attention_mask = torch.ones_like(encoded)
            return {
            "input_ids": encoded,
            "attention_mask": attention_mask,
            }

    # 某些版本会返回 BatchEncoding；这里显式转换成 dict
        return {k: torch.tensor(v) if not torch.is_tensor(v) else v for k, v in encoded.items()}
    


    @torch.no_grad()
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        enable_thinking: bool = False
    ) -> str:
        """
        生成对话回复

        Args:
            messages (List[Dict[str, str]]): 消息列表
            max_new_tokens (int, optional): 最大生成token数. Defaults to 512.
            temperature (float, optional): 采样温度. Defaults to 0.7.
            top_p (float, optional): Top-p采样. Defaults to 0.9.
            top_k (int, optional): Top-k采样. Defaults to 50.
            do_sample (bool, optional): 是否采样. Defaults to True.
            enable_thinking (bool, optional): 是否采用思考模式. Defaults to False.

        Returns:
            str: 生成的文本回复
        """
        self.model.eval()
        inputs = self.build_chat_input(messages, enable_thinking=enable_thinking)
        inputs = {
            k: v.to(self.model.device) for k, v in inputs.items()
        }

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        outputs = self.model.generate(
            **inputs,
            **generation_config,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        if enable_thinking and "</think>" in response:
            parts = response.split("</think>")
            if len(parts) > 1:
                response = parts[1].strip()
        
        return response
    
    

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
    ):
        """训练模式前向传播

        Args:
            input_ids (torch.Tensor): 输入token IDs
            attention_mask (torch.Tensor): 注意力掩码
            labels (Optional[torch.Tensor], optional): 标签(计算loss). Defaults to None.
        
        Returns:
            模型输出
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    

    
    def save_pretrained(self, save_path: str):
        """
        保存模型

        Args:
            save_path (str): 保存路径
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print("="*50)
        print(f"模型已保存到 {save_path}")
        print("="*50)
    
    
    def train_model(self):
        """
        切换到训练模式
        """
        self.model.train()
        print("="*50)
        print("模型已切换到训练模式")
    
    def eval_model(self):
        """
        切换到评估模式
        """
        self.model.eval()
        print("="*50)



# ESC专用系统提示词
ESC_SYSTEM_PROMPT = """你是一个专业的情感支持助手。你的目标是：
1. 认真倾听用户的困扰和情绪
2. 表达真诚的理解和共情
3. 使用恰当的情感支持策略（如提问、释义、肯定、建议等）
4. 帮助用户探索问题、舒缓情绪、找到解决方向

请用温暖、专业、真诚的方式与用户交流。"""


def create_esc_model(
    model_path: str= "Qwen/Qwen3-4B",
    use_lora: bool = False,
    lora_r: int = 16,
    **kwargs
) -> QwenChatModel:
    """_summary_

    Args:
        model_path (str, optional): 模型路径. Defaults to "Qwen/Qwen3-4B".
        use_lora (bool, optional): 是否应用LoRA. Defaults to False.
        lora_r (int, optional): lora rank. Defaults to 16.

    Returns:
        QwenChatModel: 配置好的QwenChatModel实例
    """
    model = QwenChatModel(
        model_path=model_path,
        **kwargs
    )
    
    if use_lora:
        model.apply_lora(r=lora_r)
    
    return model