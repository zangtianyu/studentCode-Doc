#!/usr/bin/env python3

from main import run_loss_function_comparison

if __name__ == "__main__":
    print("运行损失函数对比实验...")
    try:
        results = run_loss_function_comparison()
        print("实验完成！")
    except Exception as e:
        print(f"实验失败: {e}")
        import traceback
        traceback.print_exc()
