# 中文文章标题生成项目


## 项目结构

```
├── data/
│   ├── lcsts_test_set/     # LCSTS原始数据集
│   ├── processed/          # 预处理后的数据
|   ├── download_modelscope.py #下载LCSTS数据集
|   ├── preprocess_thucnews.py #下载LCSTS数据集,并预处理
│   └── preprocess.py       # 数据预处理脚本
├── model/
│   ├── config.py           # 模型配置
│   ├── seq2seq.py          # Baseline: Seq2Seq+LSTM+Attention
│   └── transformer.py      # 优化版: 完整Transformer
├── train/
│   ├── train_baseline.py   # Baseline训练脚本
│   ├── train_transformer.py # Transformer训练脚本
|   ├── train_mt5.py         # mt5训练脚本
|   ├── evaluate_mt5.py     # mt5测试脚本
│   └── evaluate.py         # 评估脚本   
├── utils/
│   ├── vocab.py            # 词表管理
│   ├── metrics.py          # 评估指标(ROUGE, BLEU等)
│   └── visualize.py        # 可视化工具
├── checkpoints/            # 模型检查点
├── results/                # 实验结果和图表
├── requirements.txt        # 依赖
└── README.md
```

## 环境搭建

```bash

conda create -n title_gen python=3.8
conda activate title_gen

# 安装依赖
pip install -r requirements.txt



## 快速开始

### 1. 数据预处理

**LCSTS 数据集（已下载）：**
```bash
cd d:\PythonProject\122119
python data/preprocess.py
```

**THUCNews 数据集（从 HuggingFace 下载）：**
```bash
# 首先安装 datasets 库
pip install datasets

# 预处理 THUCNews（约83万条，首次需下载2GB数据）
python data/preprocess_thucnews.py
```

### 2. 训练模型

**使用 LCSTS 数据集（默认）：**
```bash
python train/train_baseline.py --epochs 30 --batch_size 64
python train/train_transformer.py --epochs 30 --batch_size 64
python train/train_mt5.py --epochs 30 --batch_size 16 --dataset lcsts
```

**使用 THUCNews 数据集：**
```bash
python train/train_baseline.py --epochs 30 --batch_size 64 --dataset thucnews
python train/train_transformer.py --epochs 30 --batch_size 64 --dataset thucnews
```

调试模式（使用小数据集）：
```bash
python train/train_baseline.py --debug --epochs 2
```

### 3. 评估模型

```bash
# 使用 LCSTS 数据集评估
python train/evaluate.py
python train/evaluate_mt5.py

# 使用 THUCNews 数据集评估
python train/evaluate.py --dataset thucnews
```

这会在测试集上评估两个模型，输出 ROUGE、BLEU 等指标，并生成对比图。

## 实验结果

训练完成后，结果保存在 `results/` 目录：
- `baseline_loss.png`: Baseline训练曲线
- `transformer_loss.png`: Transformer训练曲线
- `model_comparison.png`: 模型指标对比图
- `metrics.json`: 详细指标数据
- `*_samples.txt`: 生成样例

