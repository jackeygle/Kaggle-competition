# ==================================================
# Jigsaw Agile Community Rules - Kaggle Notebook 版本
# 简单的 TF-IDF + 逻辑回归多标签分类模型
# ==================================================

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

print("开始训练 Jigsaw 毒性评论分类模型...")

# ==================== 1. 数据加载 ====================
print("1. 正在加载数据...")

# 读取数据文件
train_df = pd.read_csv('../input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('../input/jigsaw-agile-community-rules/test.csv')
sample_submission = pd.read_csv('../input/jigsaw-agile-community-rules/sample_submission.csv')

print(f"训练数据: {train_df.shape}")
print(f"测试数据: {test_df.shape}")
print(f"训练数据列: {list(train_df.columns)}")

# ==================== 2. 文本清洗 ====================
print("2. 正在清洗文本数据...")

def clean_text(text):
    """简单的文本清洗"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()  # 转小写
    text = re.sub(r'<[^>]+>', '', text)  # 去除HTML标签
    text = re.sub(r'[^a-zA-Z\s\.\!\?\,]', ' ', text)  # 保留字母和基本标点
    text = re.sub(r'\s+', ' ', text).strip()  # 标准化空格
    
    return text

# 应用文本清洗
train_df['comment_text_clean'] = train_df['comment_text'].apply(clean_text)
test_df['comment_text_clean'] = test_df['comment_text'].apply(clean_text)

# ==================== 3. 准备数据 ====================
print("3. 正在准备训练数据...")

# 确定目标列（根据sample_submission的列确定）
target_cols = [col for col in sample_submission.columns if col != 'id']
print(f"目标标签: {target_cols}")

# 准备特征和标签
X_text = train_df['comment_text_clean'].fillna("")
y = train_df[target_cols].values

test_text = test_df['comment_text_clean'].fillna("")
test_ids = test_df['id'].values

# 分割训练和验证集
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {len(X_train_text)}")
print(f"验证集大小: {len(X_val_text)}")

# ==================== 4. TF-IDF 特征提取 ====================
print("4. 正在提取 TF-IDF 特征...")

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(
    max_features=10000,        # 限制特征数量
    ngram_range=(1, 2),        # 使用1-gram和2-gram
    stop_words='english',      # 去除英文停用词
    lowercase=True,
    strip_accents='ascii'
)

# 拟合并转换训练数据
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf = vectorizer.transform(X_val_text)
X_test_tfidf = vectorizer.transform(test_text)

print(f"TF-IDF 特征形状: {X_train_tfidf.shape}")

# ==================== 5. 模型训练 ====================
print("5. 正在训练逻辑回归模型...")

# 创建多输出逻辑回归模型
base_model = LogisticRegression(
    C=4,                       # 正则化参数
    solver='liblinear',        # 求解器
    random_state=42,
    max_iter=1000
)

# 多标签分类器
model = MultiOutputClassifier(base_model, n_jobs=-1)

# 训练模型
model.fit(X_train_tfidf, y_train)

print("模型训练完成！")

# ==================== 6. 模型验证 ====================
print("6. 正在验证模型性能...")

# 预测验证集
y_val_pred = model.predict_proba(X_val_tfidf)

# 计算AUC分数
auc_scores = []
for i, col in enumerate(target_cols):
    if len(model.estimators_[i].classes_) == 2:
        y_pred_prob = y_val_pred[i][:, 1]  # 获取正类概率
    else:
        y_pred_prob = y_val_pred[i][:, 0]  # 如果只有一个类别
    
    auc = roc_auc_score(y_val[:, i], y_pred_prob)
    auc_scores.append(auc)
    print(f"{col} AUC: {auc:.4f}")

print(f"平均 AUC: {np.mean(auc_scores):.4f}")

# ==================== 7. 测试集预测 ====================
print("7. 正在对测试集进行预测...")

# 预测测试集
y_test_pred = model.predict_proba(X_test_tfidf)

# 整理预测结果
predictions = {'id': test_ids}
for i, col in enumerate(target_cols):
    if len(model.estimators_[i].classes_) == 2:
        predictions[col] = y_test_pred[i][:, 1]
    else:
        predictions[col] = np.zeros(len(test_ids))

# ==================== 8. 创建提交文件 ====================
print("8. 正在创建提交文件...")

# 创建提交DataFrame
submission = pd.DataFrame(predictions)

# 保存文件
submission.to_csv('submission.csv', index=False)

print("提交文件已保存！")
print("\n提交文件预览:")
print(submission.head())
print(f"\n提交文件形状: {submission.shape}")

# 显示预测统计
print("\n预测概率统计:")
for col in target_cols:
    print(f"{col}: 平均={predictions[col].mean():.4f}, 最大={predictions[col].max():.4f}")

print("\n🎉 模型训练和预测完成！可以下载 submission.csv 文件提交了！")

# ==================================================
# 可选：添加简单的数据分析
# ==================================================
print("\n" + "="*50)
print("数据分析")
print("="*50)

# 标签分布
print("训练数据标签分布:")
for col in target_cols:
    count = train_df[col].sum()
    ratio = count / len(train_df)
    print(f"{col}: {count} ({ratio:.1%})")

# 文本长度统计
text_lengths = train_df['comment_text'].str.len()
print(f"\n文本长度统计:")
print(f"平均长度: {text_lengths.mean():.1f}")
print(f"中位数长度: {text_lengths.median():.1f}")
print(f"最大长度: {text_lengths.max()}")

print("\n模型训练完成！🚀") 