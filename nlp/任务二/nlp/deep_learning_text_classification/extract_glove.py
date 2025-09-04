#!/usr/bin/env python3
"""
手动解压GloVe文件的脚本
"""

import os
import zipfile

def extract_glove():
    """解压GloVe文件"""
    glove_zip_path = "/home/zty/download/nlp/deep_learning_text_classification/embeddings/glove/glove.2024.dolma.300d.zip"
    glove_dir = "/home/zty/download/nlp/deep_learning_text_classification/embeddings/glove"
    
    print(f"检查GloVe zip文件: {glove_zip_path}")
    
    if not os.path.exists(glove_zip_path):
        print(f"❌ GloVe zip文件不存在: {glove_zip_path}")
        return False
    
    print(f"✅ 找到GloVe zip文件")
    
    # 检查文件大小
    file_size = os.path.getsize(glove_zip_path)
    print(f"文件大小: {file_size / (1024*1024):.1f} MB")
    
    # 检查是否是有效的zip文件
    try:
        with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"zip文件包含 {len(file_list)} 个文件:")
            for file_name in file_list[:5]:  # 只显示前5个
                print(f"  - {file_name}")
            if len(file_list) > 5:
                print(f"  ... 还有 {len(file_list) - 5} 个文件")
    except zipfile.BadZipFile:
        print("❌ 文件不是有效的zip文件")
        return False
    
    # 解压文件
    print("开始解压...")
    try:
        with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)
        print("✅ 解压完成")
        
        # 检查解压后的文件
        print("检查解压后的文件:")
        for dim in [50, 100, 200, 300]:
            glove_file = os.path.join(glove_dir, f"glove.6B.{dim}d.txt")
            if os.path.exists(glove_file):
                file_size = os.path.getsize(glove_file)
                print(f"  ✅ glove.6B.{dim}d.txt ({file_size / (1024*1024):.1f} MB)")
            else:
                print(f"  ❌ glove.6B.{dim}d.txt (不存在)")
        
        return True
        
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False

def main():
    """主函数"""
    print("GloVe文件解压工具")
    print("=" * 40)
    
    success = extract_glove()
    
    if success:
        print("\n🎉 GloVe文件解压成功!")
        print("现在可以运行带GloVe嵌入的训练:")
        print("python main.py --model cnn --embedding glove --epochs 5")
    else:
        print("\n⚠️ GloVe文件解压失败")
        print("建议先运行随机嵌入的训练:")
        print("python main.py --model cnn --embedding random --epochs 5")

if __name__ == "__main__":
    main()
