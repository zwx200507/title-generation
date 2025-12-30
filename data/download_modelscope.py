# -*- coding: utf-8 -*-
"""
Download complete LCSTS dataset from ModelScope (domestic mirror, fast speed)
"""

import os
import sys

def download_lcsts():
    print("正在从 ModelScope 下载 LCSTS 数据集...")
    print("（国内镜像，速度较快）")
    
    base_dir = os.path.dirname(os.path.dirname(os. path.abspath(__file__)))
    data_dir = os. path.join(base_dir, 'data', 'lcsts_full')
    
    if os.path.exists(data_dir):
        print(f"目录已存在: {data_dir}")
        return data_dir
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Use git clone to download from ModelScope
    import subprocess
    
    repo_url = "https://www.modelscope.cn/datasets/DAMO_NLP/lcsts_test_set.git"
    
    try:
        subprocess.run(
            ["git", "clone", repo_url, data_dir],
            check=True
        )
        print(f"下载完成: {data_dir}")
    except Exception as e: 
        print(f"下载失败: {e}")
        print("请手动执行:  git clone https://www.modelscope.cn/datasets/DAMO_NLP/lcsts_test_set.git")
    
    return data_dir


if __name__ == '__main__': 
    download_lcsts()