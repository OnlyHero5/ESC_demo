"""
测试Qwen3模型加载和对话功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_wrapper import QwenChatModel, ESC_SYSTEM_PROMPT
from src.utils.seed import set_seed
from src.utils.config import load_config



def test_model_loading():
    """
    测试模型加载
    """
    print("="*60)
    print("测试 1：模型加载")
    print("="*60)

    set_seed(42)

    config = load_config("configs/model.yaml")

    model = QwenChatModel(
        model_path=config["model"]["path"],
        torch_dtype=config["model"]["torch_dtype"],
        device_map=config["model"]["device_map"],
        load_in_4bit=config["model"]["load_in_4bit"],
        load_in_8bit=config["model"]["load_in_8bit"],
        use_flash_attention=config["model"]["use_flash_attention"]
    )

    print("\033[34m\n 模型加载成功！ \n\033[0m")
    return model

def test_single_turn_chat(model: QwenChatModel):
    """测试单轮对话

    Args:
        model (QwenChatModel): 模型实例
    """
    
    print("="*60)
    print("测试2： 单轮对话")
    print("="*60)

    messages = [
        {
            "role": "system", 
            "content": ESC_SYSTEM_PROMPT 
        },
        {
            "role": "user",
            "content": "我最近工作压力很大，感觉很焦虑，晚上经常失眠。"
        }
    ]

    response = model.generate_response(
        messages,
        max_new_tokens=256,
        temperature=0.7
    )
    print(f"\n用户： {messages[-1]['content']}")
    print(f"\n助手： {response}")
    print("\033[34m\n 单轮对话测试通过！ \n\033[0m")

    return response

def test_multi_turn_chat(model: QwenChatModel):
    """测试多轮对话"""
    print("="*60)
    print("测试3：多轮对话")
    print("="*60)

    messages = [
        {
            "role": "system",
            "content": ESC_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "我最近工作压力很大，感觉很焦虑。"
        },
        {
            "role": "assistant",
            "content": "我理解你现在承受着很大的工作压力，感到焦虑是很正常的反应。能告诉我是什么具体的事情让你感到压力吗？"
        },
        {
            "role": "user",
            "content": "主要是项目deadline太紧，而且老板要求很高，我怕做不好会被批评。"
        }
    ]

    response = model.generate_response(
        messages,
        max_new_tokens=256,
        temperature=0.7
    )

    print("\n对话历史:")
    for msg in messages:
        role = "系统" if msg["role"] == "system" else ("用户" if msg["role"] == "user" else "助手")
        if msg["role"]  != "system":
            print(f" {role}: {msg['content']}")
        
    print(f"\n助手（新回复）： {response}")
    print("\n 多轮对话测试通过！ \n")

def test_lora_application(model: QwenChatModel):
    """测试LoRA应用"""
    print("="*60)
    print("测试4 ： LoRA应用")
    print("="*60)

    model.apply_lora(r=16, lora_alpha=32, lora_dropout=0.05)

    messages = [
        {
            "role": "system",
            "content": ESC_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "我和我朋友吵架了，心情很低落。"
        }
    ]

    response = model.generate_response(
        messages,
        max_new_tokens=256,
        temperature=0.7,
    )
    
    print(f"\n用户: {messages[-1]['content']}")
    print(f"\n助手: {response}")
    print("\n LoRA 应用测试通过！\n")

def interactive_chat(model: QwenChatModel):
    """交互式对话测试"""
    print("="*60)
    print("交互式对话模式 quit退出")
    print("="*60)

    messages = [
        {
            "role": "system",
            "content": ESC_SYSTEM_PROMPT
        }
    ]
    print("\n开始对话\n")

    while True:
        user_input = input("你：").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n再见！")
            break
        
        if not user_input:
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        response = model.generate_response(
            messages,
            max_new_tokens=512,
            temperature=0.7
        )

        messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
        
        print(f"\n助手： {response}\n")


def main():
    print("\n" + "="*60)
    print("Qwen3 ESC 模型测试")
    print("="*60 + "\n")

    model = test_model_loading()

    test_single_turn_chat(model)

    test_multi_turn_chat(model)

    print("\n是否进入交互式对话模式？（y/n）：", end="")
    if input().strip().lower() == "y":
        interactive_chat(model)
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
