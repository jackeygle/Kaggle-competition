#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jigsaw Agile Community Rules - 增强版模型
包含高级数据清洗、交叉验证、和可视化

作者: AI Assistant
适用于: Kaggle Notebook 环境
"""

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
import nltk
import warnings
warnings.filterwarnings('ignore')

# 尝试下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        print("警告：无法下载NLTK数据，将使用基础清洗方法")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ==================== 高级数据清洗函数 ====================

def advanced_text_cleaning(text):
    """
    高级文本清洗函数
    - 去除HTML标签
    - 去除URL链接
    - 去除标点符号
    - 去除停用词
    - 去除数字
    - 转换为小写
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除URL链接
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 去除邮箱地址
    text = re.sub(r'\S+@\S+', '', text)
    
    # 转换为小写
    text = text.lower()
    
    # 去除数字
    text = re.sub(r'\d+', '', text)
    
    # 去除标点符号，但保留空格
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 尝试去除停用词
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words and len(word) > 2]
        text = ' '.join(filtered_text)
    except:
        # 如果NLTK不可用，使用简单的停用词列表
        simple_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = text.split()
        filtered_words = [word for word in words if word not in simple_stopwords and len(word) > 2]
        text = ' '.join(filtered_words)
    
    return text

def load_and_preprocess_data_enhanced():
    """
    加载并进行高级预处理
    """
    print("正在加载数据...")
    
    try:
        train_df = pd.read_csv('../input/jigsaw-agile-community-rules/train.csv')
        test_df = pd.read_csv('../input/jigsaw-agile-community-rules/test.csv')
        sample_submission = pd.read_csv('../input/jigsaw-agile-community-rules/sample_submission.csv')
    except FileNotFoundError:
        print("警告：未找到比赛数据文件，创建示例数据用于演示...")
        train_df = create_sample_data_enhanced()
        test_df = create_sample_test_data()
        sample_submission = create_sample_submission()
    
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    
    # 数据质量检查
    print("\n数据质量检查:")
    print(f"训练集缺失值: {train_df['comment_text'].isnull().sum()}")
    print(f"测试集缺失值: {test_df['comment_text'].isnull().sum()}")
    
    # 文本长度统计
    train_df['text_length'] = train_df['comment_text'].str.len()
    print(f"文本长度统计:")
    print(f"平均长度: {train_df['text_length'].mean():.2f}")
    print(f"最大长度: {train_df['text_length'].max()}")
    print(f"最小长度: {train_df['text_length'].min()}")
    
    # 高级文本清洗
    print("正在进行高级文本清洗...")
    train_df['comment_text_clean'] = train_df['comment_text'].apply(advanced_text_cleaning)
    test_df['comment_text_clean'] = test_df['comment_text'].apply(advanced_text_cleaning)
    
    # 过滤掉过短的文本
    train_df = train_df[train_df['comment_text_clean'].str.len() > 10].reset_index(drop=True)
    print(f"清洗后训练数据形状: {train_df.shape}")
    
    return train_df, test_df, sample_submission

def create_sample_data_enhanced():
    """创建增强的示例训练数据"""
    comments = [
        "This is a normal comment about the weather.",
        "You are such a stupid idiot! Go die!",
        "I disagree with your opinion but I respect your right to have it.",
        "Kill yourself you worthless piece of garbage!",
        "Nice article, thanks for sharing this information.",
        "What a beautiful day it is today!",
        "You're an absolute moron and should be banned!",
        "I found this tutorial very helpful, thank you.",
        "This is the worst article I've ever read, complete trash!",
        "Could you please provide more details about this topic?",
    ] * 100
    
    # 创建更真实的标签分布
    toxic_labels = []
    severe_toxic_labels = []
    obscene_labels = []
    threat_labels = []
    insult_labels = []
    identity_hate_labels = []
    
    for i in range(1000):
        comment_idx = i % 10
        if comment_idx in [1, 3, 6, 8]:  # 毒性评论
            toxic_labels.append(1)
            if comment_idx in [3, 6]:  # 严重毒性
                severe_toxic_labels.append(1)
            else:
                severe_toxic_labels.append(0)
            if comment_idx in [1, 8]:  # 脏话
                obscene_labels.append(1)
            else:
                obscene_labels.append(0)
            if comment_idx == 3:  # 威胁
                threat_labels.append(1)
            else:
                threat_labels.append(0)
            if comment_idx in [1, 6]:  # 侮辱
                insult_labels.append(1)
            else:
                insult_labels.append(0)
            if comment_idx == 3:  # 身份仇恨
                identity_hate_labels.append(1)
            else:
                identity_hate_labels.append(0)
        else:
            toxic_labels.append(0)
            severe_toxic_labels.append(0)
            obscene_labels.append(0)
            threat_labels.append(0)
            insult_labels.append(0)
            identity_hate_labels.append(0)
    
    data = {
        'id': range(1000),
        'comment_text': comments,
        'toxic': toxic_labels,
        'severe_toxic': severe_toxic_labels,
        'obscene': obscene_labels,
        'threat': threat_labels,
        'insult': insult_labels,
        'identity_hate': identity_hate_labels
    }
    return pd.DataFrame(data)

def create_sample_test_data():
    """创建示例测试数据"""
    test_comments = [
        "This is a test comment for evaluation.",
        "Another test comment with different content.",
        "Testing the model with this sentence.",
        "What do you think about this topic?",
        "This is an interesting discussion point."
    ] * 40
    
    data = {
        'id': range(1000, 1200),
        'comment_text': test_comments
    }
    return pd.DataFrame(data)

def create_sample_submission():
    """创建示例提交文件"""
    data = {
        'id': range(1000, 1200),
        'toxic': [0.1] * 200,
        'severe_toxic': [0.05] * 200,
        'obscene': [0.08] * 200,
        'threat': [0.03] * 200,
        'insult': [0.07] * 200,
        'identity_hate': [0.02] * 200
    }
    return pd.DataFrame(data)

# ==================== 高级特征提取 ====================

def extract_features_enhanced(train_texts, test_texts, max_features=15000):
    """
    增强的特征提取
    """
    print("正在提取增强的 TF-IDF 特征...")
    
    # 使用更复杂的TF-IDF参数
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),  # 使用1-3gram
        stop_words='english',
        lowercase=True,
        strip_accents='ascii',
        min_df=2,  # 至少出现2次
        max_df=0.95,  # 去除出现频率过高的词
        sublinear_tf=True  # 对tf使用log缩放
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    
    print(f"增强 TF-IDF 特征形状: {X_train_tfidf.shape}")
    print(f"特征维度: {X_train_tfidf.shape[1]}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer

# ==================== 交叉验证 ====================

def perform_cross_validation(X, y, target_columns, cv_folds=5):
    """
    执行交叉验证并计算F1分数
    """
    print(f"正在进行 {cv_folds} 折交叉验证...")
    
    # 创建分层K折交叉验证
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # 为每个标签存储F1分数
    f1_scores = {col: [] for col in target_columns}
    auc_scores = {col: [] for col in target_columns}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y[:, 0])):  # 使用第一个标签进行分层
        print(f"正在处理第 {fold + 1} 折...")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 训练模型
        base_classifier = LogisticRegression(
            C=4,
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        model = MultiOutputClassifier(base_classifier, n_jobs=-1)
        model.fit(X_train_fold, y_train_fold)
        
        # 预测
        y_pred = model.predict(X_val_fold)
        y_pred_proba = model.predict_proba(X_val_fold)
        
        # 计算每个标签的F1和AUC
        for i, col in enumerate(target_columns):
            f1 = f1_score(y_val_fold[:, i], y_pred[:, i])
            f1_scores[col].append(f1)
            
            if len(model.estimators_[i].classes_) == 2:
                auc = roc_auc_score(y_val_fold[:, i], y_pred_proba[i][:, 1])
            else:
                auc = 0.5  # 如果只有一个类别
            auc_scores[col].append(auc)
    
    # 输出结果
    print("\n交叉验证结果:")
    print("="*60)
    print(f"{'标签':<15} {'平均F1':<10} {'F1标准差':<10} {'平均AUC':<10} {'AUC标准差':<10}")
    print("="*60)
    
    for col in target_columns:
        f1_mean = np.mean(f1_scores[col])
        f1_std = np.std(f1_scores[col])
        auc_mean = np.mean(auc_scores[col])
        auc_std = np.std(auc_scores[col])
        print(f"{col:<15} {f1_mean:<10.4f} {f1_std:<10.4f} {auc_mean:<10.4f} {auc_std:<10.4f}")
    
    return f1_scores, auc_scores

# ==================== 可视化函数 ====================

def plot_precision_recall_curves(model, X_val, y_val, target_columns):
    """
    绘制每个类别的Precision-Recall曲线
    """
    print("正在绘制 Precision-Recall 曲线...")
    
    # 设置图形
    n_cols = 3
    n_rows = (len(target_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    y_pred_proba = model.predict_proba(X_val)
    
    for i, col in enumerate(target_columns):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row, col_idx]
        
        # 获取正类概率
        if len(model.estimators_[i].classes_) == 2:
            y_scores = y_pred_proba[i][:, 1]
        else:
            y_scores = np.zeros(len(y_val))
        
        # 计算precision和recall
        precision, recall, _ = precision_recall_curve(y_val[:, i], y_scores)
        avg_precision = average_precision_score(y_val[:, i], y_scores)
        
        # 绘制曲线
        ax.plot(recall, precision, linewidth=2, 
                label=f'AP = {avg_precision:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{col} - Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(target_columns), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_label_distribution(train_df, target_columns):
    """
    绘制标签分布图
    """
    print("正在绘制标签分布...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 标签计数
    label_counts = []
    for col in target_columns:
        label_counts.append(train_df[col].sum())
    
    # 条形图
    axes[0].bar(target_columns, label_counts)
    axes[0].set_title('标签分布 (绝对数量)')
    axes[0].set_ylabel('数量')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 比例图
    label_proportions = [count / len(train_df) for count in label_counts]
    axes[1].bar(target_columns, label_proportions)
    axes[1].set_title('标签分布 (比例)')
    axes[1].set_ylabel('比例')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== 主函数 ====================

def main_enhanced():
    """
    增强版主函数
    """
    print("="*60)
    print("Jigsaw Agile Community Rules - 增强版模型")
    print("包含高级数据清洗、交叉验证、和可视化")
    print("="*60)
    
    # 1. 加载和预处理数据
    train_df, test_df, sample_submission = load_and_preprocess_data_enhanced()
    
    # 确定目标列
    possible_targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    target_columns = [col for col in possible_targets if col in train_df.columns]
    
    if not target_columns:
        print("警告：未找到目标列，使用示例目标列")
        target_columns = possible_targets
    
    print(f"目标列: {target_columns}")
    
    # 2. 可视化标签分布
    plot_label_distribution(train_df, target_columns)
    
    # 3. 准备特征和标签
    X_text = train_df['comment_text_clean'].fillna("")
    y = train_df[target_columns].values
    test_text = test_df['comment_text_clean'].fillna("")
    test_ids = test_df['id'].values
    
    # 4. 提取增强特征
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features_enhanced(
        X_text, test_text, max_features=15000
    )
    
    # 5. 执行交叉验证
    f1_scores, auc_scores = perform_cross_validation(
        X_train_tfidf, y, target_columns, cv_folds=5
    )
    
    # 6. 训练最终模型
    print("\n正在训练最终模型...")
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_tfidf, y, test_size=0.2, random_state=42, stratify=y[:, 0]
    )
    
    base_classifier = LogisticRegression(
        C=4,
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    final_model = MultiOutputClassifier(base_classifier, n_jobs=-1)
    final_model.fit(X_train_final, y_train_final)
    
    # 7. 绘制 Precision-Recall 曲线
    plot_precision_recall_curves(final_model, X_val_final, y_val_final, target_columns)
    
    # 8. 最终预测
    print("正在对测试集进行最终预测...")
    y_pred_final = final_model.predict_proba(X_test_tfidf)
    
    predictions = {}
    for i, column in enumerate(target_columns):
        if len(final_model.estimators_[i].classes_) == 2:
            predictions[column] = y_pred_final[i][:, 1]
        else:
            predictions[column] = np.zeros(X_test_tfidf.shape[0])
    
    # 9. 创建提交文件
    submission = pd.DataFrame({'id': test_ids})
    for column in target_columns:
        submission[column] = predictions[column]
    
    submission.to_csv('submission_enhanced.csv', index=False)
    
    print("\n" + "="*60)
    print("增强版模型训练完成！")
    print("文件保存:")
    print("- submission_enhanced.csv: 提交文件")
    print("- precision_recall_curves.png: PR曲线图")
    print("- label_distribution.png: 标签分布图")
    print("="*60)
    
    # 显示最终统计
    print("\n最终预测统计:")
    for column in target_columns:
        print(f"{column}: 平均预测概率 = {predictions[column].mean():.4f}")
    
    return submission, f1_scores, auc_scores

if __name__ == "__main__":
    submission, f1_scores, auc_scores = main_enhanced() 