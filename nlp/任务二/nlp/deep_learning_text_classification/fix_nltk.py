#!/usr/bin/env python3
"""
修复NLTK问题的脚本
"""

import ssl
import nltk
import os

def fix_nltk_ssl():
    """修复NLTK的SSL证书问题"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_resources():
    """下载必要的NLTK资源"""
    fix_nltk_ssl()
    
    resources = ['punkt', 'punkt_tab', 'stopwords']
    
    for resource in resources:
        try:
            print(f"下载 {resource}...")
            nltk.download(resource, quiet=False)
            print(f"✓ {resource} 下载成功")
        except Exception as e:
            print(f"✗ {resource} 下载失败: {e}")

def test_tokenization():
    """测试分词功能"""
    print("\n测试分词功能...")
    
    # 测试简单分词
    import re
    text = "This is a test sentence with some words."
    simple_tokens = re.findall(r'\b\w+\b', text.lower())
    print(f"简单分词结果: {simple_tokens}")
    
    # 测试NLTK分词
    try:
        from nltk.tokenize import word_tokenize
        nltk_tokens = word_tokenize(text)
        print(f"NLTK分词结果: {nltk_tokens}")
    except Exception as e:
        print(f"NLTK分词失败: {e}")

def main():
    print("修复NLTK问题...")
    print("=" * 40)
    
    # 下载NLTK资源
    download_nltk_resources()
    
    # 测试分词
    test_tokenization()
    
    print("\n修复完成!")
    print("现在可以运行主程序了:")
    print("python main.py --model cnn --embedding random --epochs 5")

if __name__ == "__main__":
    main()
