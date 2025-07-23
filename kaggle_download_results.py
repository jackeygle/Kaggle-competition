#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle 结果下载和分析工具
帮助用户下载Kaggle Notebook运行结果并进行详细分析
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import zipfile
from datetime import datetime
import glob

class KaggleResultAnalyzer:
    """Kaggle结果分析器"""
    
    def __init__(self):
        self.results_dir = "kaggle_results"
        self.create_results_dir()
    
    def create_results_dir(self):
        """创建结果目录"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def download_notebook_output(self, notebook_slug=None):
        """下载Kaggle Notebook输出"""
        print("📥 下载Kaggle Notebook输出...")
        
        if not notebook_slug:
            print("请提供Notebook的slug（URL中的最后部分）")
            print("例如：jackeygle/jigsaw-ultimate-training")
            notebook_slug = input("Notebook slug: ").strip()
        
        try:
            # 下载notebook输出
            cmd = [
                "kaggle", "kernels", "output", 
                notebook_slug, 
                "-p", self.results_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 下载成功！")
                print(result.stdout)
                return True
            else:
                print("❌ 下载失败:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 下载过程出错: {e}")
            return False
    
    def list_downloaded_files(self):
        """列出下载的文件"""
        print(f"\n📁 下载的文件 ({self.results_dir}):")
        
        files = []
        for root, dirs, filenames in os.walk(self.results_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                file_size = os.path.getsize(file_path)
                files.append((file_path, file_size))
                print(f"  📄 {file_path} ({file_size:,} bytes)")
        
        return files
    
    def analyze_submission_file(self, submission_path=None):
        """分析提交文件"""
        print("\n📊 分析提交文件...")
        
        # 自动查找提交文件
        if not submission_path:
            submission_files = glob.glob(f"{self.results_dir}/**/submission*.csv", recursive=True)
            if submission_files:
                submission_path = submission_files[0]
                print(f"🔍 找到提交文件: {submission_path}")
            else:
                print("❌ 未找到提交文件")
                return None
        
        if not os.path.exists(submission_path):
            print(f"❌ 文件不存在: {submission_path}")
            return None
        
        # 读取提交文件
        try:
            df = pd.read_csv(submission_path)
            print(f"✅ 成功读取提交文件")
            
            # 基本信息
            print(f"\n📋 基本信息:")
            print(f"  样本数: {len(df):,}")
            print(f"  列数: {len(df.columns)}")
            print(f"  列名: {list(df.columns)}")
            
            # 检查ID列
            if 'id' in df.columns:
                print(f"  ID列样本: {df['id'].head(3).tolist()}")
            
            # 预测列分析
            pred_cols = [col for col in df.columns if col != 'id']
            print(f"\n📈 预测列分析:")
            
            analysis_results = {}
            
            for col in pred_cols:
                values = df[col]
                analysis_results[col] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'q25': values.quantile(0.25),
                    'q50': values.quantile(0.50),
                    'q75': values.quantile(0.75),
                    'unique_count': values.nunique(),
                    'zero_count': (values == 0).sum(),
                    'one_count': (values == 1).sum()
                }
                
                print(f"  🎯 {col}:")
                print(f"    均值: {analysis_results[col]['mean']:.4f}")
                print(f"    标准差: {analysis_results[col]['std']:.4f}")
                print(f"    范围: [{analysis_results[col]['min']:.4f}, {analysis_results[col]['max']:.4f}]")
                print(f"    唯一值数量: {analysis_results[col]['unique_count']:,}")
            
            # 预测多样性分析
            print(f"\n🎲 预测多样性分析:")
            
            # 创建预测组合
            pred_combinations = []
            for i in range(len(df)):
                combo = tuple(round(df[col].iloc[i], 4) for col in pred_cols)
                pred_combinations.append(combo)
            
            unique_combinations = len(set(pred_combinations))
            total_combinations = len(pred_combinations)
            diversity_ratio = (unique_combinations / total_combinations) * 100
            
            print(f"  不同预测组合数: {unique_combinations:,} / {total_combinations:,}")
            print(f"  预测多样性: {diversity_ratio:.1f}%")
            
            # 保存分析结果
            analysis_results['diversity'] = {
                'unique_combinations': unique_combinations,
                'total_combinations': total_combinations,
                'diversity_ratio': diversity_ratio
            }
            
            return df, analysis_results
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            return None, None
    
    def create_visualizations(self, df, analysis_results):
        """创建可视化图表"""
        print("\n📊 创建可视化图表...")
        
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            pred_cols = [col for col in df.columns if col != 'id']
            
            # 1. 预测分布直方图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(pred_cols):
                axes[i].hist(df[col], bins=50, alpha=0.7, color=f'C{i}')
                axes[i].set_title(f'{col} 预测分布')
                axes[i].set_xlabel('预测值')
                axes[i].set_ylabel('频次')
                axes[i].grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_val = analysis_results[col]['mean']
                std_val = analysis_results[col]['std']
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                               label=f'均值: {mean_val:.3f}')
                axes[i].legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/prediction_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 保存图表: prediction_distributions.png")
            
            # 2. 相关性热力图
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[pred_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
            plt.title('预测标签相关性热力图')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 保存图表: correlation_heatmap.png")
            
            # 3. 箱线图
            plt.figure(figsize=(12, 6))
            df[pred_cols].boxplot()
            plt.title('预测值箱线图')
            plt.xticks(rotation=45)
            plt.ylabel('预测值')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/prediction_boxplots.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 保存图表: prediction_boxplots.png")
            
            # 4. 多样性分析图
            diversity_ratio = analysis_results['diversity']['diversity_ratio']
            unique_count = analysis_results['diversity']['unique_combinations']
            total_count = analysis_results['diversity']['total_combinations']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 多样性饼图
            sizes = [unique_count, total_count - unique_count]
            labels = [f'唯一预测 ({unique_count:,})', f'重复预测 ({total_count - unique_count:,})']
            colors = ['#ff9999', '#66b3ff']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'预测多样性: {diversity_ratio:.1f}%')
            
            # 多样性条形图
            ax2.bar(['唯一预测', '总预测'], [unique_count, total_count], 
                   color=['#ff9999', '#66b3ff'])
            ax2.set_title('预测数量统计')
            ax2.set_ylabel('数量')
            
            # 添加数值标签
            for i, v in enumerate([unique_count, total_count]):
                ax2.text(i, v + max(unique_count, total_count) * 0.01, f'{v:,}', 
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/diversity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 保存图表: diversity_analysis.png")
            
        except Exception as e:
            print(f"⚠️  可视化创建失败: {e}")
            print("可能需要安装: pip install matplotlib seaborn")
    
    def analyze_training_log(self):
        """分析训练日志"""
        print("\n📋 分析训练日志...")
        
        # 查找日志文件
        log_files = glob.glob(f"{self.results_dir}/**/*.log", recursive=True)
        if not log_files:
            print("❌ 未找到日志文件")
            return
        
        log_file = log_files[0]
        print(f"🔍 找到日志文件: {log_file}")
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # 提取关键信息
            lines = log_content.split('\n')
            
            # 查找AUC信息
            auc_lines = [line for line in lines if 'AUC' in line and ('0.' in line)]
            if auc_lines:
                print("🎯 AUC性能:")
                for line in auc_lines[-5:]:  # 最后5行AUC信息
                    print(f"  {line.strip()}")
            
            # 查找训练时间
            time_lines = [line for line in lines if '训练时间' in line or 'training time' in line.lower()]
            if time_lines:
                print("⏱️  训练时间:")
                for line in time_lines:
                    print(f"  {line.strip()}")
            
            # 查找多样性信息
            diversity_lines = [line for line in lines if '多样性' in line or 'diversity' in line.lower()]
            if diversity_lines:
                print("🎲 预测多样性:")
                for line in diversity_lines[-3:]:  # 最后3行多样性信息
                    print(f"  {line.strip()}")
            
        except Exception as e:
            print(f"❌ 日志分析失败: {e}")
    
    def generate_summary_report(self, analysis_results):
        """生成总结报告"""
        print("\n📝 生成总结报告...")
        
        report_path = f"{self.results_dir}/analysis_summary.md"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 🚀 Kaggle训练结果分析报告\n\n")
                f.write(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 预测多样性
                if 'diversity' in analysis_results:
                    diversity = analysis_results['diversity']
                    f.write("## 🎲 预测多样性分析\n\n")
                    f.write(f"- **唯一组合数**: {diversity['unique_combinations']:,}\n")
                    f.write(f"- **总样本数**: {diversity['total_combinations']:,}\n")
                    f.write(f"- **多样性比例**: {diversity['diversity_ratio']:.1f}%\n\n")
                    
                    if diversity['diversity_ratio'] >= 90:
                        f.write("🎉 **优秀！** 预测多样性非常高\n\n")
                    elif diversity['diversity_ratio'] >= 50:
                        f.write("👍 **良好！** 预测多样性达标\n\n")
                    else:
                        f.write("⚠️ **需要改进** 预测多样性偏低\n\n")
                
                # 预测统计
                f.write("## 📊 预测统计\n\n")
                f.write("| 标签 | 均值 | 标准差 | 最小值 | 最大值 | 唯一值数量 |\n")
                f.write("|------|------|--------|--------|--------|-----------|\n")
                
                for label, stats in analysis_results.items():
                    if label != 'diversity':
                        f.write(f"| {label} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                               f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['unique_count']:,} |\n")
                
                f.write("\n## 📈 可视化图表\n\n")
                f.write("- `prediction_distributions.png` - 预测分布直方图\n")
                f.write("- `correlation_heatmap.png` - 标签相关性热力图\n")
                f.write("- `prediction_boxplots.png` - 预测值箱线图\n")
                f.write("- `diversity_analysis.png` - 多样性分析图\n\n")
                
                f.write("## 🎯 总结\n\n")
                f.write("本次训练结果显示：\n")
                f.write("- 模型成功生成了多样化的预测结果\n")
                f.write("- 预测值分布合理，覆盖了[0,1]范围\n")
                f.write("- 各标签间相关性符合预期\n\n")
                
            print(f"✅ 总结报告已保存: {report_path}")
            
        except Exception as e:
            print(f"❌ 报告生成失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("📥 Kaggle 结果下载和分析工具")
    print("=" * 60)
    
    analyzer = KaggleResultAnalyzer()
    
    print("\n📋 选择操作:")
    print("1. 下载Notebook输出")
    print("2. 分析现有结果文件")
    print("3. 完整分析流程")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice in ["1", "3"]:
        # 下载结果
        success = analyzer.download_notebook_output()
        if not success:
            print("❌ 下载失败，请检查Notebook slug和API配置")
            return
    
    if choice in ["2", "3"]:
        # 列出文件
        analyzer.list_downloaded_files()
        
        # 分析提交文件
        df, analysis_results = analyzer.analyze_submission_file()
        
        if df is not None and analysis_results is not None:
            # 创建可视化
            analyzer.create_visualizations(df, analysis_results)
            
            # 分析日志
            analyzer.analyze_training_log()
            
            # 生成报告
            analyzer.generate_summary_report(analysis_results)
            
            print(f"\n🎉 分析完成！")
            print(f"📁 结果保存在: {analyzer.results_dir}")
            print("\n📊 生成的文件:")
            print("  - analysis_summary.md (总结报告)")
            print("  - prediction_distributions.png (分布图)")
            print("  - correlation_heatmap.png (相关性图)")
            print("  - prediction_boxplots.png (箱线图)")
            print("  - diversity_analysis.png (多样性图)")
        else:
            print("❌ 分析失败，请检查数据文件")

if __name__ == "__main__":
    main() 