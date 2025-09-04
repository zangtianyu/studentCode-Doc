#!/usr/bin/env python3
"""
测试词向量加载
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_embedding_loading():
    """测试词向量加载"""
    print("测试词向量加载")
    print("=" * 40)
    
    try:
        from embeddings.embedding_loader import EmbeddingLoader
        
        # 创建加载器
        loader = EmbeddingLoader(embedding_dim=300)
        
        # 测试查找词向量文件
        glove_dir = "embeddings/glove"
        print(f"在目录中查找词向量文件: {glove_dir}")
        
        try:
            glove_path = loader.download_glove(glove_dir, dim=300)
            print(f"✅ 找到词向量文件: {glove_path}")
            
            # 测试加载少量词向量
            print("测试加载词向量...")
            embeddings = loader.load_glove_embeddings(glove_path)
            
            if embeddings:
                print(f"✅ 成功加载 {len(embeddings)} 个词向量")
                
                # 显示一些示例词向量
                sample_words = list(embeddings.keys())[:5]
                print("示例词向量:")
                for word in sample_words:
                    vector = embeddings[word]
                    print(f"  {word}: 维度={len(vector)}, 前3个值={vector[:3]}")
                
                return True
            else:
                print("❌ 未能加载任何词向量")
                return False
                
        except Exception as e:
            print(f"❌ 词向量加载失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def main():
    """主函数"""
    success = test_embedding_loading()
    
    if success:
        print("\n🎉 词向量加载测试成功!")
        print("现在可以运行完整训练:")
        print("python main.py --model cnn --embedding glove --epochs 5")
    else:
        print("\n⚠️ 词向量加载测试失败")
        print("建议使用随机嵌入:")
        print("python main.py --model cnn --embedding random --epochs 5")

if __name__ == "__main__":
    main()
