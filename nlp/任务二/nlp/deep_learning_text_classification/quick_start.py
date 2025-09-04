#!/usr/bin/env python3
"""
快速开始脚本 - 演示如何使用深度学习文本分类系统
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_requirements():
    """检查依赖包"""
    print("检查依赖包...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'scikit-learn', 
        'matplotlib', 'seaborn', 'tqdm', 'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("所有依赖包已安装!")
    return True

def check_data():
    """检查数据文件"""
    print("\n检查数据文件...")
    
    from config import DATA_DIR, TRAIN_FILE, TEST_FILE
    
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    
    if os.path.exists(train_path):
        print(f"  ✓ 训练数据: {train_path}")
        data_available = True
    else:
        print(f"  ✗ 训练数据: {train_path} (不存在)")
        data_available = False
    
    if os.path.exists(test_path):
        print(f"  ✓ 测试数据: {test_path}")
    else:
        print(f"  ✗ 测试数据: {test_path} (不存在)")
    
    if not data_available:
        print("\n数据文件不存在，请确保数据文件位于正确位置:")
        print(f"  {train_path}")
        print(f"  {test_path}")
        print("\n或者修改 config.py 中的数据路径配置")
        return False
    
    return True

def run_system_test():
    """运行系统测试"""
    print("\n运行系统测试...")
    
    try:
        from test_system import main as test_main
        return test_main()
    except Exception as e:
        print(f"系统测试失败: {e}")
        return False

def download_glove_demo():
    """演示下载GloVe词向量"""
    print("\n是否下载GloVe词向量? (y/n): ", end="")
    choice = input().lower().strip()
    
    if choice == 'y':
        print("开始下载GloVe词向量...")
        try:
            from embeddings.embedding_loader import download_and_prepare_glove
            glove_path = download_and_prepare_glove("embeddings/glove", 300)
            print(f"GloVe词向量下载完成: {glove_path}")
            return True
        except Exception as e:
            print(f"下载失败: {e}")
            print("可以稍后使用以下命令下载:")
            print("python main.py --mode download_glove")
            return False
    else:
        print("跳过GloVe下载")
        return True

def run_quick_training():
    """运行快速训练演示"""
    print("\n运行快速训练演示...")
    print("使用CNN模型 + 随机嵌入，训练5个epoch")
    
    try:
        import subprocess
        import sys
        
        # 运行训练命令
        cmd = [
            sys.executable, "main.py",
            "--model", "cnn",
            "--embedding", "random",
            "--epochs", "5",
            "--batch_size", "32"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("快速训练完成!")
            return True
        else:
            print("训练过程中出现错误")
            return False
            
    except Exception as e:
        print(f"训练失败: {e}")
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\n" + "=" * 50)
    print("使用示例")
    print("=" * 50)
    
    examples = [
        ("训练CNN模型 + 随机嵌入", "python main.py --model cnn --embedding random"),
        ("训练RNN模型 + GloVe嵌入", "python main.py --model rnn --embedding glove"),
        ("自定义训练参数", "python main.py --model cnn --embedding glove --epochs 20 --batch_size 64 --lr 0.001"),
        ("运行模型对比实验", "python main.py --mode compare"),
        ("运行嵌入方式对比", "python main.py --mode embedding_compare"),
        ("运行所有实验", "python main.py --mode all"),
        ("下载GloVe词向量", "python main.py --mode download_glove"),
        ("系统测试", "python test_system.py")
    ]
    
    for description, command in examples:
        print(f"\n{description}:")
        print(f"  {command}")
    
    print(f"\n更多选项请查看:")
    print(f"  python main.py --help")

def main():
    """主函数"""
    print("🚀 深度学习文本分类系统 - 快速开始")
    print("=" * 50)
    
    # 1. 检查依赖
    if not check_requirements():
        return
    
    # 2. 检查数据
    data_ok = check_data()
    
    # 3. 运行系统测试
    if data_ok:
        print("\n是否运行系统测试? (y/n): ", end="")
        choice = input().lower().strip()
        
        if choice == 'y':
            test_ok = run_system_test()
            if not test_ok:
                print("系统测试未完全通过，但可以继续尝试")
    
    # 4. GloVe下载演示
    if data_ok:
        download_glove_demo()
    
    # 5. 快速训练演示
    if data_ok:
        print("\n是否运行快速训练演示? (y/n): ", end="")
        choice = input().lower().strip()
        
        if choice == 'y':
            run_quick_training()
    
    # 6. 显示使用示例
    show_usage_examples()
    
    print("\n🎉 快速开始完成!")
    print("现在你可以开始使用深度学习文本分类系统了!")

if __name__ == "__main__":
    main()
