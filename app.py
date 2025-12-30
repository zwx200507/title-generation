# -*- coding: utf-8 -*-
"""
Title generation Web application
Based on Flask + mT5 model
"""

import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

app = Flask(__name__)

# Global variables to store model
model = None
tokenizer = None
device = None


def load_model():
    """Load model"""
    global model, tokenizer, device
    
    device = torch.device('cuda' if torch. cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'checkpoints', 'mt5_best')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    
    print("正在加载模型...")
    tokenizer = MT5Tokenizer. from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print("✓ 模型加载成功")


def generate_title(text):
    """Generate title"""
    if model is None or tokenizer is None:
        return "模型未加载"
    
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
    
    title = tokenizer.decode(generated[0], skip_special_tokens=True)
    return title


@app. route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate title API"""
    try: 
        data = request.get_json()
        text = data. get('text', '').strip()
        
        if not text: 
            return jsonify({'error':  '请输入文本内容'}), 400
        
        if len(text) > 1000:
            return jsonify({'error': '文本长度不能超过1000字符'}), 400
        
        title = generate_title(text)
        return jsonify({'title': title})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Start Flask application
    app.run(host='0.0.0.0', port=5000, debug=False)