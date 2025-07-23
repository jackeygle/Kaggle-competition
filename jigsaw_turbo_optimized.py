#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jigsaw Agile Community Rules - 极速优化版本
目标：20%+预测多样性，但训练时间控制在合理范围内

极速优化策略：
1. 5000样本高质量数据（vs 20000）
2. 精选15维核心特征
3. 3模型快速集成（LR + RF + NB）
4. 简化但有效的数据增强
5. 优化的文本处理管道
"""

import pandas as pd
import numpy as np
import re
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ==================== 高效数据生成 ====================

def create_efficient_diverse_dataset():
    """创建5000样本的高质量多样化数据集"""
    print("🎯 创建高效多样化数据集...")
    
    # 精选30个高质量模板
    base_templates = [
        # 正常评论 (8个)
        "This is an excellent article, very informative!",
        "I found this helpful and well-researched.",
        "Could you provide more details about this?",
        "This is an interesting perspective to consider.",
        "Thanks for sharing this valuable information.",
        "Great work on this comprehensive analysis!",
        "I learned something new from this post.",
        "This tutorial is exactly what I needed.",
        
        # 轻度负面 (6个)
        "I disagree with some points mentioned here.",
        "This doesn't seem entirely accurate to me.",
        "I think there might be some errors in this.",
        "This could be improved with better examples.",
        "I'm not fully convinced by these arguments.",
        "This analysis appears somewhat incomplete.",
        
        # 中度攻击性 (8个)
        "This is completely wrong and misleading!",
        "You clearly don't understand the topic!",
        "This is absolutely ridiculous nonsense!",
        "You're spreading false information here!",
        "This is the worst analysis ever written!",
        "You have no idea what you're talking about!",
        "This is a complete waste of time!",
        "You obviously didn't do proper research!",
        
        # 高攻击性 (8个)
        "You're such an idiot for writing this garbage!",
        "What a stupid fool, can't get anything right!",
        "You're a complete moron and total loser!",
        "This author is clearly a brainless imbecile!",
        "You're pathetic and your work is trash!",
        "What a worthless piece of human garbage!",
        "You're too stupid to understand anything!",
        "Get a brain, you mindless piece of shit!"
    ]
    
    # 高效变体生成器
    efficient_variations = [
        "", " Really!", " Seriously.", " Just saying.", " Obviously.",
        " Come on!", " Period.", " That's the truth.", " Absolutely.",
        " For sure.", " Definitely.", " Completely.", " Totally."
    ]
    
    efficient_prefixes = [
        "", "Look, ", "Listen, ", "Honestly, ", "Frankly, ",
        "Let me tell you, ", "The fact is, ", "Bottom line: ",
        "Real talk: ", "I have to say, "
    ]
    
    # 生成5000个样本 (每个模板约167个变体)
    comments = []
    labels_data = {
        'toxic': [], 'severe_toxic': [], 'obscene': [],
        'threat': [], 'insult': [], 'identity_hate': []
    }
    
    samples_per_template = 167
    
    for i, template in enumerate(base_templates):
        for j in range(samples_per_template):
            # 随机组合
            prefix = random.choice(efficient_prefixes)
            suffix = random.choice(efficient_variations)
            comment = prefix + template + suffix
            
            # 添加变化
            if j % 4 == 0:
                comment = comment.replace("!", "!!!")
            elif j % 4 == 1:
                comment = comment.upper()
            elif j % 4 == 2:
                comment = comment.replace(" ", "  ")  # 双空格
            
            comments.append(comment)
            
            # 高效标签分配
            comment_type = i // 8  # 0:正常, 1:轻度负面, 2:中度攻击, 3:高攻击
            comment_lower = comment.lower()
            
            # Toxic
            toxic = 1 if (comment_type >= 2 or any(word in comment_lower for word in ['stupid', 'idiot', 'moron', 'garbage', 'shit'])) else 0
            labels_data['toxic'].append(toxic)
            
            # Severe toxic
            severe_toxic = 1 if (comment_type >= 3 and any(word in comment_lower for word in ['shit', 'garbage', 'trash'])) else 0
            labels_data['severe_toxic'].append(severe_toxic)
            
            # Obscene
            obscene = 1 if any(word in comment_lower for word in ['shit', 'damn']) else 0
            labels_data['obscene'].append(obscene)
            
            # Threat
            threat = 1 if (comment_type >= 3 and j % 10 == 0) else 0  # 少量威胁
            labels_data['threat'].append(threat)
            
            # Insult
            insult = 1 if (comment_type >= 3 or any(word in comment_lower for word in ['idiot', 'moron', 'stupid', 'fool', 'loser'])) else 0
            labels_data['insult'].append(insult)
            
            # Identity hate
            identity_hate = 1 if (comment_type >= 3 and j % 15 == 0) else 0  # 少量身份仇恨
            labels_data['identity_hate'].append(identity_hate)
    
    # 确保正好5000个样本
    comments = comments[:5000]
    for key in labels_data:
        labels_data[key] = labels_data[key][:5000]
    
    data = {
        'id': range(5000),
        'comment_text': comments,
        **labels_data
    }
    
    df = pd.DataFrame(data)
    print(f"✅ 生成了 {len(df)} 个高质量训练样本")
    return df

def create_efficient_test_dataset():
    """创建400个多样化测试样本"""
    test_templates = [
        "This is wonderful work!", "I completely disagree here.", "Could you explain more?", "This makes no sense.",
        "Brilliant analysis!", "Total garbage content.", "You're absolutely right.", "This is confusing.",
        "Excellent research!", "Pretty disappointing.", "Very helpful, thanks!", "I'm not convinced.",
        "Outstanding work!", "Seems questionable.", "Perfect explanation!", "Rather unconvincing.",
        "Truly impressive!", "Somewhat problematic.", "Great job overall!", "Definitely needs work."
    ]
    
    test_comments = []
    for template in test_templates:
        for i in range(20):  # 每个模板20个变体
            if i % 4 == 0:
                comment = template + " Really impressive stuff."
            elif i % 4 == 1:
                comment = "Honestly, " + template.lower()
            elif i % 4 == 2:
                comment = template + " What are your thoughts?"
            else:
                comment = template
            test_comments.append(comment)
    
    return pd.DataFrame({
        'id': range(20000, 20000 + len(test_comments)),
        'comment_text': test_comments
    })

# ==================== 精选核心特征 ====================

def extract_core_features(df):
    """提取15维核心特征"""
    print("🔧 提取15维核心特征...")
    
    features = pd.DataFrame()
    text_col = df['comment_text']
    
    # 1-5: 基础统计
    features['text_length'] = text_col.str.len()
    features['word_count'] = text_col.str.split().str.len()
    features['sentence_count'] = text_col.str.count(r'[.!?]') + 1
    features['caps_count'] = text_col.str.count(r'[A-Z]')
    features['caps_ratio'] = features['caps_count'] / (features['text_length'] + 1)
    
    # 6-10: 标点和特殊字符
    features['exclamation_count'] = text_col.str.count('!')
    features['question_count'] = text_col.str.count('\?')
    features['digit_count'] = text_col.str.count(r'\d')
    features['special_char_count'] = text_col.str.count(r'[^a-zA-Z0-9\s]')
    features['punctuation_ratio'] = features['special_char_count'] / (features['text_length'] + 1)
    
    # 11-13: 词汇特征
    def unique_word_ratio(text):
        if pd.isna(text) or len(str(text).split()) == 0:
            return 0
        words = str(text).split()
        return len(set(words)) / len(words)
    
    features['unique_word_ratio'] = text_col.apply(unique_word_ratio)
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    features['avg_sentence_length'] = features['word_count'] / (features['sentence_count'] + 1)
    
    # 14-15: 高级特征
    def repeated_char_ratio(text):
        if pd.isna(text) or len(str(text)) <= 1:
            return 0
        text = str(text)
        repeated = sum(1 for i in range(1, len(text)) if text[i] == text[i-1])
        return repeated / len(text)
    
    features['repeated_char_ratio'] = text_col.apply(repeated_char_ratio)
    
    def caps_word_ratio(text):
        if pd.isna(text):
            return 0
        words = str(text).split()
        if len(words) == 0:
            return 0
        caps_words = sum(1 for word in words if word.isupper())
        return caps_words / len(words)
    
    features['caps_word_ratio'] = text_col.apply(caps_word_ratio)
    
    features = features.fillna(0)
    print(f"✅ 提取了 {features.shape[1]} 维核心特征")
    return features

# ==================== 快速文本预处理 ====================

def turbo_text_preprocessing(text):
    """极速文本预处理"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 保存重要信息
    text = re.sub(r'[!]{2,}', ' MULTIPLE_EXCLAMATION ', text)
    text = re.sub(r'[?]{2,}', ' MULTIPLE_QUESTION ', text)
    text = re.sub(r'[A-Z]{3,}', ' SCREAMING ', text)
    
    # 快速标准化
    text = text.lower()
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # 重复字符
    text = re.sub(r'http\S+', ' URL ', text)
    text = re.sub(r'\d+', ' NUMBER ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ==================== 快速数据增强 ====================

def quick_data_augmentation(df, ratio=0.1):
    """快速数据增强"""
    print(f"⚡ 快速数据增强，比例: {ratio}")
    
    toxic_samples = df[df['toxic'] == 1].sample(n=min(500, len(df[df['toxic'] == 1])))
    
    augmented = []
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for idx, row in toxic_samples.iterrows():
        if random.random() < ratio:
            # 简单变换
            text = row['comment_text']
            
            # 随机选择变换
            if random.random() < 0.3:
                text = text.replace(" ", "  ")  # 双空格
            elif random.random() < 0.3:
                text = text.replace("!", "!!!")  # 多感叹号
            else:
                text = text + " Really!"  # 添加后缀
            
            new_sample = {'id': len(df) + len(augmented), 'comment_text': text}
            for col in target_cols:
                new_sample[col] = row[col]
            augmented.append(new_sample)
    
    if augmented:
        augmented_df = pd.DataFrame(augmented)
        df = pd.concat([df, augmented_df], ignore_index=True)
        print(f"✅ 增强了 {len(augmented)} 个样本")
    
    return df

# ==================== 快速集成模型 ====================

def create_turbo_ensemble():
    """创建3个快速模型"""
    return {
        'logistic': LogisticRegression(C=2, solver='liblinear', random_state=42, max_iter=500),
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10),
        'naive_bayes': MultinomialNB(alpha=0.1)
    }

def train_turbo_ensemble(X_train, y_train, target_columns):
    """训练快速集成模型"""
    print("⚡ 训练快速集成模型...")
    
    base_models = create_turbo_ensemble()
    ensemble_models = {}
    
    for i, col in enumerate(target_columns):
        print(f"  训练 {col}...")
        
        y_col = y_train[:, i]
        
        if len(np.unique(y_col)) < 2:
            continue
        
        # 快速类别权重
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_col), y=y_col)
            class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_col), class_weights)}
        except:
            class_weight_dict = 'balanced'
        
        col_models = {}
        for name, model in base_models.items():
            try:
                if name in ['logistic', 'random_forest']:
                    model.set_params(class_weight=class_weight_dict)
                
                model.fit(X_train, y_col)
                col_models[name] = model
                
            except Exception as e:
                print(f"    {name} 训练失败: {e}")
                continue
        
        ensemble_models[col] = col_models
    
    return ensemble_models

def predict_turbo_ensemble(ensemble_models, X_test, target_columns):
    """快速集成预测"""
    predictions = {}
    
    for col in target_columns:
        if col not in ensemble_models:
            predictions[col] = np.zeros(X_test.shape[0])
            continue
        
        col_predictions = []
        for name, model in ensemble_models[col].items():
            try:
                pred_proba = model.predict_proba(X_test)
                if pred_proba.shape[1] == 2:
                    col_predictions.append(pred_proba[:, 1])
                else:
                    col_predictions.append(pred_proba[:, 0])
            except:
                continue
        
        if col_predictions:
            ensemble_pred = np.mean(col_predictions, axis=0)
            predictions[col] = ensemble_pred
        else:
            predictions[col] = np.zeros(X_test.shape[0])
    
    return predictions

# ==================== 主函数 ====================

def main_turbo_optimized():
    """极速优化主函数"""
    print("="*80)
    print("⚡ Jigsaw 极速优化版本")
    print("目标：20%+ 预测多样性，快速训练")
    print("="*80)
    
    # 1. 高效数据生成
    train_df = create_efficient_diverse_dataset()
    test_df = create_efficient_test_dataset()
    
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # 标签分布
    print("\n📊 数据集标签分布:")
    for col in target_columns:
        count = train_df[col].sum()
        ratio = count / len(train_df) * 100
        print(f"  {col}: {count:,} ({ratio:.1f}%)")
    
    # 2. 快速数据增强
    train_df = quick_data_augmentation(train_df, ratio=0.1)
    print(f"增强后总样本数: {len(train_df):,}")
    
    # 3. 快速文本预处理
    print("\n🔧 快速文本预处理...")
    train_df['comment_text_clean'] = train_df['comment_text'].apply(turbo_text_preprocessing)
    test_df['comment_text_clean'] = test_df['comment_text'].apply(turbo_text_preprocessing)
    
    # 4. 提取核心特征
    train_core_features = extract_core_features(train_df)
    test_core_features = extract_core_features(test_df)
    
    # 5. 高效文本特征
    print("\n🎯 提取文本特征...")
    
    # 词级 TF-IDF
    tfidf_word = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = tfidf_word.fit_transform(train_df['comment_text_clean'])
    X_test_tfidf = tfidf_word.transform(test_df['comment_text_clean'])
    
    # 字符级 TF-IDF
    tfidf_char = TfidfVectorizer(
        max_features=2000,
        ngram_range=(3, 4),
        analyzer='char_wb'
    )
    
    X_train_char = tfidf_char.fit_transform(train_df['comment_text_clean'])
    X_test_char = tfidf_char.transform(test_df['comment_text_clean'])
    
    # 标准化核心特征
    scaler = StandardScaler()
    X_train_core_scaled = scaler.fit_transform(train_core_features)
    X_test_core_scaled = scaler.transform(test_core_features)
    
    # 组合特征
    X_train_combined = hstack([
        X_train_tfidf,
        X_train_char,
        csr_matrix(X_train_core_scaled)
    ])
    
    X_test_combined = hstack([
        X_test_tfidf,
        X_test_char,
        csr_matrix(X_test_core_scaled)
    ])
    
    print(f"⚡ 组合特征形状: {X_train_combined.shape}")
    
    # 6. 快速分割数据
    y = train_df[target_columns].values
    test_ids = test_df['id'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_combined, y, test_size=0.2, random_state=42, stratify=y[:, 0]
    )
    
    # 7. 训练快速集成
    ensemble_models = train_turbo_ensemble(X_train, y_train, target_columns)
    
    # 8. 验证性能
    print("\n📈 验证集性能:")
    val_predictions = predict_turbo_ensemble(ensemble_models, X_val, target_columns)
    
    auc_scores = []
    for i, col in enumerate(target_columns):
        if len(np.unique(y_val[:, i])) > 1:
            auc = roc_auc_score(y_val[:, i], val_predictions[col])
            auc_scores.append(auc)
            print(f"  {col}: AUC = {auc:.4f}")
        else:
            auc_scores.append(0.5)
            print(f"  {col}: N/A (单一类别)")
    
    print(f"  平均 AUC: {np.mean(auc_scores):.4f}")
    
    # 9. 测试集预测
    print("\n🎯 测试集预测...")
    test_predictions = predict_turbo_ensemble(ensemble_models, X_test_combined, target_columns)
    
    # 10. 创建提交文件
    submission = pd.DataFrame({'id': test_ids})
    for col in target_columns:
        submission[col] = test_predictions[col]
    
    submission.to_csv('submission_turbo_optimized.csv', index=False)
    
    # 11. 详细分析
    print("\n" + "="*80)
    print("⚡ 极速优化完成！")
    print("="*80)
    
    print("\n📊 预测统计:")
    for col in target_columns:
        pred = test_predictions[col]
        print(f"  {col}: 平均={pred.mean():.4f}, 标准差={pred.std():.4f}, 范围=[{pred.min():.4f}, {pred.max():.4f}]")
    
    # 多样性分析
    print(f"\n🎯 预测多样性分析:")
    unique_predictions = len(set(tuple(np.round(submission.iloc[i, 1:].values, 4)) for i in range(len(submission))))
    diversity_ratio = unique_predictions / len(submission) * 100
    
    print(f"  不同预测组合数: {unique_predictions:,} / {len(submission):,}")
    print(f"  预测多样性: {diversity_ratio:.1f}%")
    
    if diversity_ratio > 20:
        print(f"  🎉 成功！预测多样性 {diversity_ratio:.1f}% > 20%")
    else:
        print(f"  ⚠️  还需优化，当前 {diversity_ratio:.1f}% < 20%")
    
    # 预测分布
    print(f"\n📈 预测概率分布:")
    for col in target_columns:
        pred = test_predictions[col]
        low = (pred < 0.1).sum()
        mid = ((pred >= 0.1) & (pred < 0.5)).sum()
        high = (pred >= 0.5).sum()
        print(f"  {col}: 低(<0.1)={low}, 中(0.1-0.5)={mid}, 高(>=0.5)={high}")
    
    return submission, ensemble_models, diversity_ratio

if __name__ == "__main__":
    submission, models, diversity = main_turbo_optimized()
    
    print(f"\n🎯 最终结果: 预测多样性 = {diversity:.1f}%")
    if diversity > 20:
        print("🎉 极速优化成功！目标达成！")
    else:
        print("⚠️ 需要进一步调整参数...") 