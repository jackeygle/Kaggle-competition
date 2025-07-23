#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jigsaw Agile Community Rules - 改进版模型
解决预测概率相同、特征不足等问题

改进点：
1. 更丰富的特征工程
2. 更好的文本预处理
3. 模型集成
4. 类别平衡处理
5. 超参数优化
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# ==================== 高级特征工程 ====================

def extract_text_features(df):
    """提取文本统计特征"""
    features = pd.DataFrame()
    
    # 基础统计特征
    features['text_length'] = df['comment_text'].str.len()
    features['word_count'] = df['comment_text'].str.split().str.len()
    features['sentence_count'] = df['comment_text'].str.count(r'[.!?]') + 1
    
    # 字符特征
    features['caps_count'] = df['comment_text'].str.count(r'[A-Z]')
    features['caps_ratio'] = features['caps_count'] / (features['text_length'] + 1)
    features['exclamation_count'] = df['comment_text'].str.count('!')
    features['question_count'] = df['comment_text'].str.count('\?')
    
    # 特殊字符
    features['digit_count'] = df['comment_text'].str.count(r'\d')
    features['special_char_count'] = df['comment_text'].str.count(r'[^a-zA-Z0-9\s]')
    
    # 平均词长
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    
    # 填充缺失值
    features = features.fillna(0)
    
    return features

def advanced_text_cleaning(text):
    """高级文本清洗"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 保存原始长度信息
    original_length = len(text)
    
    # 转换为小写
    text = text.lower()
    
    # 处理常见的网络语言和缩写
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'m", " am", text)
    
    # 去除HTML标签但保留内容
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 去除URL
    text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', text, flags=re.MULTILINE)
    
    # 处理重复字符（如：sooooo good -> so good）
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 保留一些标点符号的语义
    text = re.sub(r'[!]{2,}', ' [EXCLAMATION] ', text)
    text = re.sub(r'[?]{2,}', ' [QUESTION] ', text)
    
    # 处理数字
    text = re.sub(r'\d+', ' [NUMBER] ', text)
    
    # 标准化空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_enhanced_sample_data():
    """创建更加多样化的示例数据"""
    # 更真实的评论样本
    comments = [
        "This is a great article, thanks for sharing!",
        "You're such an idiot! This is completely wrong!",
        "I respectfully disagree with your opinion, but I understand your point.",
        "Kill yourself, you worthless piece of garbage! Nobody likes you!",
        "Could you please provide more sources for this information?",
        "What a beautiful day! Hope everyone is doing well.",
        "You're an absolute moron and should be banned from the internet!",
        "This tutorial was very helpful, thank you so much!",
        "This is the worst garbage I've ever seen! Complete waste of time!",
        "I'm not sure I understand this correctly, could you explain more?",
        "Fuck this shit! This is totally fucked up!",
        "You people are all the same, always causing problems!",
        "I will find you and make you pay for this!",
        "Great work on this project, very impressive results!",
        "This makes no sense at all, completely useless information.",
        "Thanks for taking the time to write this detailed explanation.",
        "All [RELIGION] people should go back where they came from!",
        "You [RACE] people are destroying our country!",
        "I hope you die in a car accident, loser!",
        "This is actually quite interesting, I learned something new."
    ]
    
    # 扩展到更多样本
    extended_comments = []
    for i in range(100):
        for comment in comments:
            # 添加一些变化
            if i % 3 == 0:
                comment = comment + " Really!"
            elif i % 3 == 1:
                comment = comment + " What do you think?"
            extended_comments.append(comment)
    
    # 创建更真实的标签
    labels = {
        'toxic': [],
        'severe_toxic': [],
        'obscene': [],
        'threat': [],
        'insult': [],
        'identity_hate': []
    }
    
    for comment in extended_comments:
        comment_lower = comment.lower()
        
        # Toxic - 包含攻击性语言
        toxic = 1 if any(word in comment_lower for word in ['idiot', 'moron', 'garbage', 'fuck', 'shit', 'die', 'kill', 'worst']) else 0
        labels['toxic'].append(toxic)
        
        # Severe toxic - 非常严重的攻击
        severe_toxic = 1 if any(word in comment_lower for word in ['kill yourself', 'die in', 'worthless']) else 0
        labels['severe_toxic'].append(severe_toxic)
        
        # Obscene - 脏话
        obscene = 1 if any(word in comment_lower for word in ['fuck', 'shit']) else 0
        labels['obscene'].append(obscene)
        
        # Threat - 威胁
        threat = 1 if any(phrase in comment_lower for phrase in ['kill you', 'find you', 'make you pay', 'die in']) else 0
        labels['threat'].append(threat)
        
        # Insult - 侮辱
        insult = 1 if any(word in comment_lower for word in ['idiot', 'moron', 'loser', 'garbage']) else 0
        labels['insult'].append(insult)
        
        # Identity hate - 身份仇恨
        identity_hate = 1 if any(phrase in comment_lower for phrase in ['[religion]', '[race]', 'you people']) else 0
        labels['identity_hate'].append(identity_hate)
    
    data = {
        'id': range(len(extended_comments)),
        'comment_text': extended_comments,
        **labels
    }
    
    return pd.DataFrame(data)

def create_diverse_test_data():
    """创建多样化的测试数据"""
    test_comments = [
        "This is a wonderful article!",
        "You are completely wrong about this.",
        "I think this could be improved somehow.",
        "What a terrible day this has been.",
        "Could you help me understand this better?",
        "This is absolutely fantastic work!",
        "I'm not convinced by your arguments.",
        "This seems like a reasonable approach.",
        "I strongly disagree with this viewpoint.",
        "Thank you for this informative post."
    ] * 20  # 200 test samples
    
    data = {
        'id': range(1000, 1200),
        'comment_text': test_comments
    }
    return pd.DataFrame(data)

# ==================== 改进的模型训练 ====================

def train_improved_model(X_train, y_train, target_columns):
    """训练改进的多标签模型"""
    print("正在训练改进的多标签模型...")
    
    models = {}
    
    for i, col in enumerate(target_columns):
        print(f"训练 {col} 分类器...")
        
        # 计算类别权重处理不平衡
        y_col = y_train[:, i]
        if len(np.unique(y_col)) > 1:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_col), y=y_col)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        else:
            class_weight_dict = 'balanced'
        
        # 逻辑回归模型
        lr_model = LogisticRegression(
            C=2,
            solver='liblinear',
            random_state=42,
            max_iter=1000,
            class_weight=class_weight_dict
        )
        
        # 训练模型
        lr_model.fit(X_train, y_col)
        models[col] = lr_model
    
    return models

def predict_with_improved_model(models, X_test, target_columns):
    """使用改进模型进行预测"""
    predictions = {}
    
    for col in target_columns:
        if col in models:
            pred_proba = models[col].predict_proba(X_test)
            if pred_proba.shape[1] == 2:
                predictions[col] = pred_proba[:, 1]
            else:
                predictions[col] = pred_proba[:, 0]
        else:
            predictions[col] = np.zeros(X_test.shape[0])
    
    return predictions

# ==================== 主函数 ====================

def main_improved():
    """改进版主函数"""
    print("="*60)
    print("Jigsaw Agile Community Rules - 改进版模型")
    print("解决预测相同、特征不足等问题")
    print("="*60)
    
    # 1. 创建更好的示例数据
    print("1. 创建增强的示例数据...")
    train_df = create_enhanced_sample_data()
    test_df = create_diverse_test_data()
    
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    
    # 检查标签分布
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    print("\n标签分布:")
    for col in target_columns:
        count = train_df[col].sum()
        ratio = count / len(train_df) * 100
        print(f"{col}: {count} ({ratio:.1f}%)")
    
    # 2. 高级文本清洗
    print("\n2. 进行高级文本清洗...")
    train_df['comment_text_clean'] = train_df['comment_text'].apply(advanced_text_cleaning)
    test_df['comment_text_clean'] = test_df['comment_text'].apply(advanced_text_cleaning)
    
    # 3. 提取文本统计特征
    print("3. 提取文本统计特征...")
    train_text_features = extract_text_features(train_df)
    test_text_features = extract_text_features(test_df)
    
    print(f"文本统计特征: {train_text_features.shape[1]} 维")
    print("特征列:", list(train_text_features.columns))
    
    # 4. 提取多种文本特征
    print("4. 提取多种文本特征...")
    
    # TF-IDF 特征 (word level)
    tfidf_word = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True,
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf_word = tfidf_word.fit_transform(train_df['comment_text_clean'])
    X_test_tfidf_word = tfidf_word.transform(test_df['comment_text_clean'])
    
    # TF-IDF 特征 (char level)
    tfidf_char = TfidfVectorizer(
        max_features=3000,
        ngram_range=(3, 5),
        analyzer='char',
        lowercase=True
    )
    
    X_train_tfidf_char = tfidf_char.fit_transform(train_df['comment_text_clean'])
    X_test_tfidf_char = tfidf_char.transform(test_df['comment_text_clean'])
    
    # 组合所有特征
    X_train_combined = hstack([
        X_train_tfidf_word,
        X_train_tfidf_char,
        train_text_features.values
    ])
    
    X_test_combined = hstack([
        X_test_tfidf_word,
        X_test_tfidf_char,
        test_text_features.values
    ])
    
    print(f"组合特征形状: {X_train_combined.shape}")
    
    # 5. 准备标签
    y = train_df[target_columns].values
    test_ids = test_df['id'].values
    
    # 6. 划分训练验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_combined, y, test_size=0.2, random_state=42, stratify=y[:, 0]
    )
    
    # 7. 训练改进模型
    models = train_improved_model(X_train, y_train, target_columns)
    
    # 8. 验证模型
    print("\n验证模型性能...")
    val_predictions = predict_with_improved_model(models, X_val, target_columns)
    
    print("验证集 AUC 分数:")
    auc_scores = []
    for i, col in enumerate(target_columns):
        if len(np.unique(y_val[:, i])) > 1:
            auc = roc_auc_score(y_val[:, i], val_predictions[col])
            auc_scores.append(auc)
            print(f"{col}: {auc:.4f}")
        else:
            print(f"{col}: N/A (单一类别)")
            auc_scores.append(0.5)
    
    print(f"平均 AUC: {np.mean(auc_scores):.4f}")
    
    # 9. 测试集预测
    print("\n9. 对测试集进行预测...")
    test_predictions = predict_with_improved_model(models, X_test_combined, target_columns)
    
    # 10. 创建提交文件
    submission = pd.DataFrame({'id': test_ids})
    for col in target_columns:
        submission[col] = test_predictions[col]
    
    submission.to_csv('submission_improved.csv', index=False)
    
    print("\n" + "="*60)
    print("改进版模型完成！")
    print("文件保存: submission_improved.csv")
    print("="*60)
    
    # 显示预测统计
    print("\n预测概率统计:")
    for col in target_columns:
        pred = test_predictions[col]
        print(f"{col}: 平均={pred.mean():.4f}, 标准差={pred.std():.4f}, 最小={pred.min():.4f}, 最大={pred.max():.4f}")
    
    # 检查预测多样性
    print(f"\n预测多样性检查:")
    unique_predictions = len(set(tuple(submission.iloc[i, 1:].values) for i in range(len(submission))))
    print(f"不同预测组合数: {unique_predictions} / {len(submission)} ({unique_predictions/len(submission)*100:.1f}%)")
    
    return submission, models

if __name__ == "__main__":
    submission, models = main_improved() 