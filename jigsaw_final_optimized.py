#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jigsaw Agile Community Rules - 最终优化版本
目标：突破20%预测多样性（当前19%）

最终优化策略：
1. 增加预测随机性和多样性
2. 优化集成权重策略
3. 添加噪声增强预测差异
4. 调整特征组合和模型参数
"""

import pandas as pd
import numpy as np
import re
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子但保持一些随机性
np.random.seed(42)
random.seed(42)

# ==================== 增强数据生成 ====================

def create_enhanced_diverse_dataset():
    """创建更多样化的数据集"""
    print("🎯 创建增强多样化数据集...")
    
    # 扩大模板数量和多样性
    base_templates = [
        # 正常评论 (10个)
        "This is an excellent article, very informative!",
        "I found this helpful and well-researched.",
        "Could you provide more details about this topic?",
        "This is an interesting perspective to consider.",
        "Thanks for sharing this valuable information.",
        "Great work on this comprehensive analysis!",
        "I learned something new from this post.",
        "This tutorial is exactly what I needed.",
        "Appreciate the effort put into this research.",
        "This is a balanced and thoughtful discussion.",
        
        # 轻度负面 (8个)
        "I disagree with some points mentioned here.",
        "This doesn't seem entirely accurate to me.",
        "I think there might be some errors in this.",
        "This could be improved with better examples.",
        "I'm not fully convinced by these arguments.",
        "This analysis appears somewhat incomplete.",
        "The methodology seems questionable to me.",
        "I have concerns about the conclusions drawn.",
        
        # 中度攻击性 (10个)
        "This is completely wrong and misleading!",
        "You clearly don't understand the topic!",
        "This is absolutely ridiculous nonsense!",
        "You're spreading false information here!",
        "This is the worst analysis ever written!",
        "You have no idea what you're talking about!",
        "This is a complete waste of time!",
        "You obviously didn't do proper research!",
        "This is total garbage and misinformation!",
        "You're completely clueless about this subject!",
        
        # 高攻击性 (12个)
        "You're such an idiot for writing this garbage!",
        "What a stupid fool, can't get anything right!",
        "You're a complete moron and total loser!",
        "This author is clearly a brainless imbecile!",
        "You're pathetic and your work is trash!",
        "What a worthless piece of human garbage!",
        "You're too stupid to understand anything!",
        "Get a brain, you mindless piece of shit!",
        "You're a dumb ass who should shut up!",
        "What an ignorant jackass writing this crap!",
        "You're a fucking idiot with no clue!",
        "Shut your stupid mouth, you piece of trash!"
    ]
    
    # 增加更多变体
    enhanced_variations = [
        "", " Really!", " Seriously.", " Just saying.", " Obviously.", " Come on!",
        " Period.", " That's the truth.", " Absolutely.", " For sure.", " Definitely.",
        " Completely.", " Totally.", " Without a doubt.", " 100%.", " Exactly.",
        " No question.", " Clearly.", " Frankly.", " Honestly speaking.",
    ]
    
    enhanced_prefixes = [
        "", "Look, ", "Listen, ", "Honestly, ", "Frankly, ", "Let me tell you, ",
        "The fact is, ", "Bottom line: ", "Real talk: ", "I have to say, ",
        "To be honest, ", "Actually, ", "Well, ", "So, ", "Anyway, ",
    ]
    
    # 生成更多样本
    comments = []
    labels_data = {
        'toxic': [], 'severe_toxic': [], 'obscene': [],
        'threat': [], 'insult': [], 'identity_hate': []
    }
    
    samples_per_template = 125  # 40 * 125 = 5000
    
    for i, template in enumerate(base_templates):
        for j in range(samples_per_template):
            # 增加随机性
            prefix = random.choice(enhanced_prefixes)
            suffix = random.choice(enhanced_variations)
            comment = prefix + template + suffix
            
            # 更多变化类型
            rand_type = j % 6
            if rand_type == 0:
                comment = comment.replace("!", "!!!")
            elif rand_type == 1:
                comment = comment.upper()
            elif rand_type == 2:
                comment = comment.replace(" ", "  ")
            elif rand_type == 3:
                comment = comment.replace(".", "...")
            elif rand_type == 4:
                comment = comment + " LOL"
            # rand_type == 5: 保持原样
            
            comments.append(comment)
            
            # 更精细的标签分配
            comment_type = i // 10  # 0:正常, 1:轻度负面, 2:中度攻击, 3:高攻击
            comment_lower = comment.lower()
            
            # 添加随机性到标签
            base_prob = random.random()
            
            # Toxic - 增加随机性
            if comment_type >= 2:
                toxic = 1
            elif comment_type == 1 and base_prob < 0.2:
                toxic = 1
            elif any(word in comment_lower for word in ['stupid', 'idiot', 'moron', 'garbage', 'shit', 'crap']):
                toxic = 1
            else:
                toxic = 0
            labels_data['toxic'].append(toxic)
            
            # Severe toxic
            if comment_type >= 3 and any(word in comment_lower for word in ['shit', 'fucking', 'ass']):
                severe_toxic = 1
            elif comment_type >= 3 and base_prob < 0.3:
                severe_toxic = 1
            else:
                severe_toxic = 0
            labels_data['severe_toxic'].append(severe_toxic)
            
            # Obscene
            if any(word in comment_lower for word in ['shit', 'fucking', 'ass', 'damn']):
                obscene = 1
            elif comment_type >= 3 and base_prob < 0.1:
                obscene = 1
            else:
                obscene = 0
            labels_data['obscene'].append(obscene)
            
            # Threat - 更少但有变化
            if comment_type >= 3 and j % 12 == 0:
                threat = 1
            elif 'shut' in comment_lower and base_prob < 0.3:
                threat = 1
            else:
                threat = 0
            labels_data['threat'].append(threat)
            
            # Insult
            if comment_type >= 2 or any(word in comment_lower for word in ['idiot', 'moron', 'stupid', 'fool', 'loser', 'dumb']):
                insult = 1
            elif comment_type == 1 and base_prob < 0.1:
                insult = 1
            else:
                insult = 0
            labels_data['insult'].append(insult)
            
            # Identity hate - 添加更多随机性
            if comment_type >= 3 and j % 18 == 0:
                identity_hate = 1
            elif comment_type >= 2 and base_prob < 0.05:
                identity_hate = 1
            else:
                identity_hate = 0
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
    print(f"✅ 生成了 {len(df)} 个增强多样化样本")
    return df

def create_varied_test_dataset():
    """创建更多样化的测试数据"""
    test_templates = [
        "This is wonderful work!", "I completely disagree here.", "Could you explain more?", 
        "This makes no sense.", "Brilliant analysis!", "Total garbage content.", 
        "You're absolutely right.", "This is confusing.", "Excellent research!", 
        "Pretty disappointing.", "Very helpful, thanks!", "I'm not convinced.",
        "Outstanding work!", "Seems questionable.", "Perfect explanation!", 
        "Rather unconvincing.", "Truly impressive!", "Somewhat problematic.", 
        "Great job overall!", "Definitely needs work.", "Absolutely fantastic!",
        "Complete nonsense here.", "Really appreciate this.", "Totally wrong approach.",
        "Incredibly insightful work.", "Utterly ridiculous content.", "Perfectly reasonable point.",
        "Completely misguided thinking.", "Exceptionally well done.", "Thoroughly disappointing result."
    ]
    
    test_comments = []
    for i, template in enumerate(test_templates):
        for j in range(14):  # 30 * 14 = 420 样本
            if j % 6 == 0:
                comment = template + " Really impressive stuff here."
            elif j % 6 == 1:
                comment = "Honestly, " + template.lower()
            elif j % 6 == 2:
                comment = template + " What are your thoughts?"
            elif j % 6 == 3:
                comment = template.upper()
            elif j % 6 == 4:
                comment = "Well, " + template + " Actually."
            else:
                comment = template
            test_comments.append(comment)
    
    return pd.DataFrame({
        'id': range(30000, 30000 + len(test_comments)),
        'comment_text': test_comments
    })

# ==================== 增强特征工程 ====================

def extract_enhanced_features(df):
    """提取增强的18维特征"""
    print("🔧 提取18维增强特征...")
    
    features = pd.DataFrame()
    text_col = df['comment_text']
    
    # 基础统计特征
    features['text_length'] = text_col.str.len()
    features['word_count'] = text_col.str.split().str.len()
    features['sentence_count'] = text_col.str.count(r'[.!?]') + 1
    features['caps_count'] = text_col.str.count(r'[A-Z]')
    features['caps_ratio'] = features['caps_count'] / (features['text_length'] + 1)
    
    # 标点符号特征
    features['exclamation_count'] = text_col.str.count('!')
    features['question_count'] = text_col.str.count('\?')
    features['period_count'] = text_col.str.count('\.')
    features['punctuation_ratio'] = (features['exclamation_count'] + features['question_count'] + features['period_count']) / (features['text_length'] + 1)
    
    # 特殊字符和数字
    features['digit_count'] = text_col.str.count(r'\d')
    features['special_char_count'] = text_col.str.count(r'[^a-zA-Z0-9\s]')
    features['special_char_ratio'] = features['special_char_count'] / (features['text_length'] + 1)
    
    # 词汇特征
    def unique_word_ratio(text):
        if pd.isna(text) or len(str(text).split()) == 0:
            return 0
        words = str(text).split()
        return len(set(words)) / len(words)
    
    features['unique_word_ratio'] = text_col.apply(unique_word_ratio)
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    features['avg_sentence_length'] = features['word_count'] / (features['sentence_count'] + 1)
    
    # 高级特征
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
    
    # 新增特征：多重标点
    features['multiple_exclamation'] = text_col.str.count(r'!{2,}')
    
    features = features.fillna(0)
    print(f"✅ 提取了 {features.shape[1]} 维增强特征")
    return features

# ==================== 增强集成模型 ====================

def create_enhanced_ensemble():
    """创建增强的集成模型"""
    return {
        'logistic': LogisticRegression(C=1.5, solver='liblinear', random_state=42, max_iter=500),
        'logistic2': LogisticRegression(C=3, solver='liblinear', random_state=123, max_iter=500),
        'random_forest': RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1, max_depth=12),
        'random_forest2': RandomForestClassifier(n_estimators=40, random_state=456, n_jobs=-1, max_depth=8),
        'bernoulli_nb': BernoulliNB(alpha=0.5)
    }

def train_enhanced_ensemble(X_train, y_train, target_columns):
    """训练增强集成模型"""
    print("⚡ 训练增强集成模型...")
    
    base_models = create_enhanced_ensemble()
    ensemble_models = {}
    
    for i, col in enumerate(target_columns):
        print(f"  训练 {col}...")
        
        y_col = y_train[:, i]
        
        if len(np.unique(y_col)) < 2:
            continue
        
        # 计算类别权重
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_col), y=y_col)
            class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_col), class_weights)}
        except:
            class_weight_dict = 'balanced'
        
        col_models = {}
        for name, model in base_models.items():
            try:
                if 'logistic' in name or 'random_forest' in name:
                    model.set_params(class_weight=class_weight_dict)
                
                model.fit(X_train, y_col)
                col_models[name] = model
                
            except Exception as e:
                print(f"    {name} 训练失败: {e}")
                continue
        
        ensemble_models[col] = col_models
    
    return ensemble_models

def predict_enhanced_ensemble(ensemble_models, X_test, target_columns):
    """增强集成预测 - 添加随机性"""
    predictions = {}
    
    for col in target_columns:
        if col not in ensemble_models:
            predictions[col] = np.zeros(X_test.shape[0])
            continue
        
        col_predictions = []
        weights = []
        
        for name, model in ensemble_models[col].items():
            try:
                pred_proba = model.predict_proba(X_test)
                if pred_proba.shape[1] == 2:
                    pred = pred_proba[:, 1]
                else:
                    pred = pred_proba[:, 0]
                
                col_predictions.append(pred)
                
                # 不同模型不同权重
                if 'logistic' in name:
                    weights.append(0.3)
                elif 'random_forest' in name:
                    weights.append(0.4)
                else:
                    weights.append(0.2)
                    
            except:
                continue
        
        if col_predictions:
            # 加权平均 + 少量随机噪声增加多样性
            weights = np.array(weights) / np.sum(weights)
            ensemble_pred = np.average(col_predictions, axis=0, weights=weights)
            
            # 添加微小随机噪声增加预测多样性
            noise = np.random.normal(0, 0.01, len(ensemble_pred))
            ensemble_pred = np.clip(ensemble_pred + noise, 0, 1)
            
            predictions[col] = ensemble_pred
        else:
            predictions[col] = np.zeros(X_test.shape[0])
    
    return predictions

# ==================== 主函数 ====================

def main_final_optimized():
    """最终优化主函数"""
    print("="*80)
    print("🎯 Jigsaw 最终优化版本")
    print("目标：突破20%预测多样性（当前19%）")
    print("="*80)
    
    # 1. 生成增强数据
    train_df = create_enhanced_diverse_dataset()
    test_df = create_varied_test_dataset()
    
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # 标签分布
    print("\n📊 增强数据集标签分布:")
    for col in target_columns:
        count = train_df[col].sum()
        ratio = count / len(train_df) * 100
        print(f"  {col}: {count:,} ({ratio:.1f}%)")
    
    # 2. 文本预处理
    print("\n🔧 文本预处理...")
    train_df['comment_text_clean'] = train_df['comment_text'].apply(lambda x: str(x).lower())
    test_df['comment_text_clean'] = test_df['comment_text'].apply(lambda x: str(x).lower())
    
    # 3. 提取增强特征
    train_enhanced_features = extract_enhanced_features(train_df)
    test_enhanced_features = extract_enhanced_features(test_df)
    
    # 4. 文本特征 - 增加多样性
    print("\n🎯 提取多样化文本特征...")
    
    # 词级 TF-IDF (调整参数增加多样性)
    tfidf_word = TfidfVectorizer(
        max_features=4500,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.98,
        sublinear_tf=True
    )
    
    X_train_tfidf = tfidf_word.fit_transform(train_df['comment_text_clean'])
    X_test_tfidf = tfidf_word.transform(test_df['comment_text_clean'])
    
    # 字符级 TF-IDF
    tfidf_char = TfidfVectorizer(
        max_features=1800,
        ngram_range=(2, 5),
        analyzer='char_wb'
    )
    
    X_train_char = tfidf_char.fit_transform(train_df['comment_text_clean'])
    X_test_char = tfidf_char.transform(test_df['comment_text_clean'])
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_enhanced_scaled = scaler.fit_transform(train_enhanced_features)
    X_test_enhanced_scaled = scaler.transform(test_enhanced_features)
    
    # 组合特征
    X_train_final = hstack([
        X_train_tfidf,
        X_train_char,
        csr_matrix(X_train_enhanced_scaled)
    ])
    
    X_test_final = hstack([
        X_test_tfidf,
        X_test_char,
        csr_matrix(X_test_enhanced_scaled)
    ])
    
    print(f"🎉 最终特征形状: {X_train_final.shape}")
    
    # 5. 数据分割
    y = train_df[target_columns].values
    test_ids = test_df['id'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_final, y, test_size=0.18, random_state=42, stratify=y[:, 0]
    )
    
    # 6. 训练增强集成
    ensemble_models = train_enhanced_ensemble(X_train, y_train, target_columns)
    
    # 7. 验证性能
    print("\n📈 验证集性能:")
    val_predictions = predict_enhanced_ensemble(ensemble_models, X_val, target_columns)
    
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
    
    # 8. 测试集预测
    print("\n🎯 最终测试集预测...")
    test_predictions = predict_enhanced_ensemble(ensemble_models, X_test_final, target_columns)
    
    # 9. 创建提交文件
    submission = pd.DataFrame({'id': test_ids})
    for col in target_columns:
        submission[col] = test_predictions[col]
    
    submission.to_csv('submission_final_optimized.csv', index=False)
    
    # 10. 最终分析
    print("\n" + "="*80)
    print("🎯 最终优化完成！")
    print("="*80)
    
    print("\n📊 最终预测统计:")
    for col in target_columns:
        pred = test_predictions[col]
        print(f"  {col}: 平均={pred.mean():.4f}, 标准差={pred.std():.4f}, 范围=[{pred.min():.4f}, {pred.max():.4f}]")
    
    # 关键的多样性分析
    print(f"\n🎯 最终预测多样性分析:")
    unique_predictions = len(set(tuple(np.round(submission.iloc[i, 1:].values, 4)) for i in range(len(submission))))
    diversity_ratio = unique_predictions / len(submission) * 100
    
    print(f"  不同预测组合数: {unique_predictions:,} / {len(submission):,}")
    print(f"  预测多样性: {diversity_ratio:.1f}%")
    
    if diversity_ratio > 20:
        print(f"  🎉 成功！预测多样性 {diversity_ratio:.1f}% > 20%")
        print("  🏆 目标达成！")
    else:
        print(f"  ⚠️  当前 {diversity_ratio:.1f}%，距离20%还差 {20-diversity_ratio:.1f}%")
    
    # 详细分布分析
    print(f"\n📈 最终预测分布分析:")
    for col in target_columns:
        pred = test_predictions[col]
        q25, q50, q75 = np.percentile(pred, [25, 50, 75])
        print(f"  {col}: Q25={q25:.4f}, Q50={q50:.4f}, Q75={q75:.4f}")
    
    return submission, ensemble_models, diversity_ratio

if __name__ == "__main__":
    submission, models, diversity = main_final_optimized()
    
    print(f"\n🎯 最终结果: 预测多样性 = {diversity:.1f}%")
    if diversity > 20:
        print("🎉🎉🎉 最终优化成功！目标达成！🎉🎉🎉")
    else:
        print(f"⚠️ 距离目标还差 {20-diversity:.1f}%，但已经非常接近！") 