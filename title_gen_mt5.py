# -*- coding: utf-8 -*-
"""
Interactive title generation program
User inputs a piece of text, mT5 model automatically generates title
"""

import os
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer


def generate_title(text, model, tokenizer, device):
    """Generate title using mT5 model"""
    model.eval()
    
    # Encode input text
    inputs = tokenizer(
        text, max_length=256, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        generated = model. generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode generated title
    title = tokenizer.decode(generated[0], skip_special_tokens=True)
    return title


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # Load model
    script_dir = os. path.dirname(os.path. abspath(__file__))
    model_path = os.path. join(script_dir, 'checkpoints', 'mt5_best')
    
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return
    
    print("正在加载模型...")
    tokenizer = MT5Tokenizer. from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print("✓ 模型加载成功\n")
    
    # Interactive interface
    print("="*60)
    print("交互式标题生成程序")
    print("="*60)
    print("输入文本内容，程序将自动生成标题")
    print("输入 'quit' 或 'exit' 退出程序")
    print("="*60 + "\n")
    
    while True: 
        try:
            user_input = input("请输入文本: ").strip()
            
            if not user_input: 
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！👋")
                break
            
            # Generate title
            print("\n生成中...")
            title = generate_title(user_input, model, tokenizer, device)
            print(f"📌 生成标题:  {title}\n")
        
        except KeyboardInterrupt: 
            print("\n\n程序被中断，再见！👋")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")
            continue


if __name__ == '__main__':
    main()