#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jigsaw Agile Community Rules - 入门模型
使用 TF-IDF + 逻辑回归进行毒性评论多标签分类

作者: AI Assistant
适用于: Kaggle Notebook 环境
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据预处理函数 ====================

def clean_text(text):
    """
    清洗文本数据
    - 转换为小写
    - 去除特殊字符和数字
    - 去除多余空格
    """
    if pd.isna(text):
        return ""
    
    # 转换为小写
    text = str(text).lower()
    
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 保留字母、空格和基本标点符号
    text = re.sub(r'[^a-zA-Z\s\.\!\?\,]', ' ', text)
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_preprocess_data():
    """
    加载并预处理数据
    """
    print("正在加载数据...")
    
    # 在 Kaggle 环境中，数据路径通常是 ../input/competition-name/
    # 这里使用通用路径，你可能需要根据实际比赛调整
    try:
        # 尝试加载训练数据
        train_df = pd.read_csv('../input/jigsaw-agile-community-rules/train.csv')
        test_df = pd.read_csv('../input/jigsaw-agile-community-rules/test.csv')
        sample_submission = pd.read_csv('../input/jigsaw-agile-community-rules/sample_submission.csv')
    except FileNotFoundError:
        # 如果找不到文件，创建示例数据用于测试
        print("警告：未找到比赛数据文件，创建示例数据用于演示...")
        train_df = create_sample_data()
        test_df = create_sample_test_data()
        sample_submission = create_sample_submission()
    
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    
    # 检查数据列
    print("训练数据列:", train_df.columns.tolist())
    
    # 清洗文本数据
    print("正在清洗文本数据...")
    train_df['comment_text_clean'] = train_df['comment_text'].apply(clean_text)
    test_df['comment_text_clean'] = test_df['comment_text'].apply(clean_text)
    
    return train_df, test_df, sample_submission

def create_sample_data():
    """创建示例训练数据"""
    data = {
        'id': range(1000),
        'comment_text': [
            "This is a normal comment.",
            "You are so stupid and ugly!",
            "I disagree with your opinion but respect it.",
            "Go kill yourself, loser!",
            "Nice article, thanks for sharing.",
        ] * 200,
        'toxic': [0, 1, 0, 1, 0] * 200,
        'severe_toxic': [0, 0, 0, 1, 0] * 200,
        'obscene': [0, 1, 0, 1, 0] * 200,
        'threat': [0, 0, 0, 1, 0] * 200,
        'insult': [0, 1, 0, 1, 0] * 200,
        'identity_hate': [0, 0, 0, 1, 0] * 200
    }
    return pd.DataFrame(data)

def create_sample_test_data():
    """创建示例测试数据"""
    data = {
        'id': range(1000, 1200),
        'comment_text': [
            "This is a test comment.",
            "Another test comment here.",
        ] * 100
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

# ==================== 特征提取 ====================

def extract_features(train_texts, test_texts, max_features=10000):
    """
    使用 TF-IDF 提取文本特征
    
    参数:
    - train_texts: 训练集文本
    - test_texts: 测试集文本
    - max_features: 最大特征数量
    
    返回:
    - X_train_tfidf: 训练集 TF-IDF 特征
    - X_test_tfidf: 测试集 TF-IDF 特征
    """
    print("正在提取 TF-IDF 特征...")
    
    # 创建 TF-IDF 向量化器
    # ngram_range=(1,2) 表示使用 unigram 和 bigram
    # max_features 限制特征数量避免内存问题
    # stop_words='english' 去除英文停用词
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True,
        strip_accents='ascii'
    )
    
    # 拟合训练数据并转换
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    
    # 转换测试数据
    X_test_tfidf = vectorizer.transform(test_texts)
    
    print(f"TF-IDF 特征形状: {X_train_tfidf.shape}")
    print(f"特征维度: {X_train_tfidf.shape[1]}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer

# ==================== 模型训练 ====================

def train_model(X_train, y_train, target_columns):
    """
    训练多标签逻辑回归模型
    
    参数:
    - X_train: 训练特征
    - y_train: 训练标签
    - target_columns: 目标列名
    
    返回:
    - model: 训练好的模型
    """
    print("正在训练多标签逻辑回归模型...")
    
    # 使用 MultiOutputClassifier 包装逻辑回归进行多标签分类
    # C=4 是正则化参数，solver='liblinear' 适合小数据集
    base_classifier = LogisticRegression(
        C=4,
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    
    # 多输出分类器
    model = MultiOutputClassifier(base_classifier, n_jobs=-1)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    print("模型训练完成！")
    
    return model

# ==================== 模型评估 ====================

def evaluate_model(model, X_val, y_val, target_columns):
    """
    评估模型性能
    """
    print("正在评估模型...")
    
    # 预测验证集
    y_pred = model.predict_proba(X_val)
    
    # 计算每个标签的 AUC
    auc_scores = []
    for i, column in enumerate(target_columns):
        # 获取正类概率
        if len(model.estimators_[i].classes_) == 2:
            y_pred_prob = y_pred[i][:, 1]
        else:
            y_pred_prob = y_pred[i][:, 0]
        
        auc = roc_auc_score(y_val[:, i], y_pred_prob)
        auc_scores.append(auc)
        print(f"{column} AUC: {auc:.4f}")
    
    overall_auc = np.mean(auc_scores)
    print(f"平均 AUC: {overall_auc:.4f}")
    
    return overall_auc

# ==================== 预测和提交 ====================

def make_predictions(model, X_test, target_columns):
    """
    对测试集进行预测
    """
    print("正在对测试集进行预测...")
    
    # 预测概率
    y_pred = model.predict_proba(X_test)
    
    # 整理预测结果
    predictions = {}
    for i, column in enumerate(target_columns):
        # 获取正类概率
        if len(model.estimators_[i].classes_) == 2:
            predictions[column] = y_pred[i][:, 1]
        else:
            # 如果只有一个类别，预测概率为 0
            predictions[column] = np.zeros(X_test.shape[0])
    
    return predictions

def create_submission(test_ids, predictions, target_columns, filename='submission.csv'):
    """
    创建提交文件
    """
    print("正在创建提交文件...")
    
    # 创建提交 DataFrame
    submission = pd.DataFrame({'id': test_ids})
    
    # 添加预测结果
    for column in target_columns:
        submission[column] = predictions[column]
    
    # 保存文件
    submission.to_csv(filename, index=False)
    
    print(f"提交文件已保存为: {filename}")
    print("提交文件前几行:")
    print(submission.head())
    print(f"提交文件形状: {submission.shape}")
    
    return submission

# ==================== 主函数 ====================

def main():
    """
    主函数：执行完整的机器学习流程
    """
    print("="*50)
    print("Jigsaw Agile Community Rules - 入门模型")
    print("="*50)
    
    # 1. 加载和预处理数据
    train_df, test_df, sample_submission = load_and_preprocess_data()
    
    # 确定目标列（根据实际数据调整）
    possible_targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    target_columns = [col for col in possible_targets if col in train_df.columns]
    
    if not target_columns:
        print("警告：未找到目标列，使用示例目标列")
        target_columns = possible_targets
    
    print(f"目标列: {target_columns}")
    
    # 2. 准备特征和标签
    X_text = train_df['comment_text_clean'].fillna("")
    y = train_df[target_columns].values
    
    test_text = test_df['comment_text_clean'].fillna("")
    test_ids = test_df['id'].values
    
    # 3. 划分训练集和验证集
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y[:, 0]
    )
    
    print(f"训练集大小: {len(X_train_text)}")
    print(f"验证集大小: {len(X_val_text)}")
    
    # 4. 提取 TF-IDF 特征
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(
        X_train_text, test_text, max_features=10000
    )
    X_val_tfidf = vectorizer.transform(X_val_text)
    
    # 5. 训练模型
    model = train_model(X_train_tfidf, y_train, target_columns)
    
    # 6. 评估模型
    evaluate_model(model, X_val_tfidf, y_val, target_columns)
    
    # 7. 对测试集进行预测
    predictions = make_predictions(model, X_test_tfidf, target_columns)
    
    # 8. 创建提交文件
    submission = create_submission(test_ids, predictions, target_columns)
    
    print("="*50)
    print("任务完成！")
    print("现在你可以下载 submission.csv 文件并提交到 Kaggle！")
    print("="*50)
    
    # 显示一些统计信息
    print("\n预测统计:")
    for column in target_columns:
        print(f"{column}: 平均预测概率 = {predictions[column].mean():.4f}")

if __name__ == "__main__":
    main() 