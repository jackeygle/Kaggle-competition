#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Notebook 上传和管理工具
帮助用户将代码上传到Kaggle，并管理运行结果
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
import subprocess
import time

class KaggleNotebookManager:
    """Kaggle Notebook管理器"""
    
    def __init__(self):
        self.project_name = "jigsaw-ultimate-solution"
        self.notebook_title = "Jigsaw Ultimate Training Script"
        
    def create_kaggle_dataset(self):
        """创建Kaggle数据集来存储代码"""
        print("🚀 创建Kaggle数据集...")
        
        # 创建数据集目录
        dataset_dir = "kaggle_dataset"
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)
        
        # 复制代码文件
        files_to_upload = [
            "jigsaw_ultimate_kaggle_script.py",
            "jigsaw_kaggle_test.py", 
            "requirements_ultimate.txt",
            "KAGGLE_ULTIMATE_GUIDE.md"
        ]
        
        for file in files_to_upload:
            if os.path.exists(file):
                shutil.copy2(file, dataset_dir)
                print(f"✅ 复制文件: {file}")
        
        # 创建数据集元数据
        dataset_metadata = {
            "title": "Jigsaw Ultimate Training Scripts",
            "id": f"jackeygle/{self.project_name}",
            "licenses": [{"name": "MIT"}],
            "resources": []
        }
        
        # 添加文件资源
        for file in files_to_upload:
            if os.path.exists(file):
                dataset_metadata["resources"].append({
                    "path": file,
                    "description": f"Training script: {file}"
                })
        
        # 保存元数据
        with open(f"{dataset_dir}/dataset-metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print(f"✅ 数据集创建完成: {dataset_dir}")
        return dataset_dir
    
    def upload_to_kaggle(self, dataset_dir):
        """上传数据集到Kaggle"""
        print("📤 上传数据集到Kaggle...")
        
        try:
            # 切换到数据集目录
            original_dir = os.getcwd()
            os.chdir(dataset_dir)
            
            # 创建新数据集
            result = subprocess.run([
                "kaggle", "datasets", "create", "-p", "."
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 数据集上传成功！")
                print(result.stdout)
            else:
                print("⚠️  数据集可能已存在，尝试更新...")
                # 尝试更新现有数据集
                update_result = subprocess.run([
                    "kaggle", "datasets", "version", "-p", ".", "-m", "Updated training scripts"
                ], capture_output=True, text=True)
                
                if update_result.returncode == 0:
                    print("✅ 数据集更新成功！")
                    print(update_result.stdout)
                else:
                    print("❌ 上传失败:")
                    print(update_result.stderr)
            
            # 回到原目录
            os.chdir(original_dir)
            
        except Exception as e:
            print(f"❌ 上传过程出错: {e}")
            os.chdir(original_dir)
    
    def create_notebook_template(self):
        """创建Kaggle Notebook模板"""
        print("📝 创建Notebook模板...")
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🚀 Jigsaw Ultimate Training Script\n",
                        "\n",
                        "这是一个高性能的多标签分类训练脚本，目标AUC ≥ 0.99\n",
                        "\n",
                        "## 📊 特性\n",
                        "- ✅ 自动优化循环\n",
                        "- ✅ 多模型集成\n", 
                        "- ✅ GPU加速支持\n",
                        "- ✅ 详细日志输出\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# 导入代码文件\n",
                        "import sys\n",
                        "sys.path.append('../input/jigsaw-ultimate-solution')\n",
                        "\n",
                        "# 检查GPU可用性\n",
                        "import torch\n",
                        "print(f\"GPU可用: {torch.cuda.is_available()}\")\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f\"GPU型号: {torch.cuda.get_device_name()}\")\n",
                        "    print(f\"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# 方案1: 运行轻量级版本（推荐）\n",
                        "exec(open('../input/jigsaw-ultimate-solution/jigsaw_kaggle_test.py').read())"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# 方案2: 运行终极版（需要安装额外依赖）\n",
                        "# !pip install transformers torch torchvision\n",
                        "# exec(open('../input/jigsaw-ultimate-solution/jigsaw_ultimate_kaggle_script.py').read())"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# 检查输出文件\n",
                        "import os\n",
                        "import pandas as pd\n",
                        "\n",
                        "# 列出输出文件\n",
                        "print(\"📁 输出文件:\")\n",
                        "for file in os.listdir('/kaggle/working'):\n",
                        "    print(f\"  - {file}\")\n",
                        "\n",
                        "# 检查提交文件\n",
                        "if os.path.exists('/kaggle/working/submission_ultimate.csv'):\n",
                        "    submission = pd.read_csv('/kaggle/working/submission_ultimate.csv')\n",
                        "    print(f\"\\n📊 提交文件统计:\")\n",
                        "    print(f\"  样本数: {len(submission)}\")\n",
                        "    print(f\"  列数: {len(submission.columns)}\")\n",
                        "    print(f\"  列名: {list(submission.columns)}\")\n",
                        "    print(f\"\\n📈 预测统计:\")\n",
                        "    for col in submission.columns[1:]:\n",
                        "        mean_val = submission[col].mean()\n",
                        "        std_val = submission[col].std()\n",
                        "        print(f\"  {col}: 均值={mean_val:.4f}, 标准差={std_val:.4f}\")"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.7.12"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # 保存notebook
        with open("kaggle_notebook.ipynb", "w") as f:
            json.dump(notebook_content, f, indent=2)
        
        print("✅ Notebook模板创建完成: kaggle_notebook.ipynb")
        return "kaggle_notebook.ipynb"

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Kaggle Notebook 上传工具")
    print("=" * 60)
    
    manager = KaggleNotebookManager()
    
    # 检查Kaggle API
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        print(f"✅ Kaggle API版本: {result.stdout.strip()}")
    except:
        print("❌ Kaggle API未安装或未配置")
        print("请运行: pip install kaggle")
        print("并配置API密钥")
        return
    
    print("\n📋 选择操作:")
    print("1. 创建并上传数据集")
    print("2. 创建Notebook模板")
    print("3. 全部执行")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice in ["1", "3"]:
        # 创建和上传数据集
        dataset_dir = manager.create_kaggle_dataset()
        manager.upload_to_kaggle(dataset_dir)
        
        print(f"\n🎯 数据集链接:")
        print(f"https://www.kaggle.com/datasets/jackeygle/{manager.project_name}")
    
    if choice in ["2", "3"]:
        # 创建Notebook模板
        notebook_file = manager.create_notebook_template()
        print(f"\n📝 请手动上传notebook文件: {notebook_file}")
        print("上传地址: https://www.kaggle.com/code")
    
    print("\n✅ 操作完成！")
    
    print("\n📋 下一步操作:")
    print("1. 访问: https://www.kaggle.com/code")
    print("2. 点击 'New Notebook'")
    print("3. 上传生成的 kaggle_notebook.ipynb")
    print("4. 在 'Data' 标签页添加数据集: jackeygle/jigsaw-ultimate-solution")
    print("5. 开启GPU: Settings → Accelerator → GPU T4 x2")
    print("6. 运行所有代码块")
    print("7. 点击 'Save Version' 保存结果")

if __name__ == "__main__":
    main() 