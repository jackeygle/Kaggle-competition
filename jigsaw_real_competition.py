#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jigsaw Agile Community Rules - 真实比赛数据训练脚本
🎯 目标：在真实Kaggle比赛数据上训练模型

比赛信息：
- 任务：二分类（预测是否违反社区规则）
- 训练数据：2029个样本
- 测试数据：67个样本
- 特征：body(文本) + rule(规则) + subreddit(社区) + 示例
"""

import pandas as pd
import numpy as np
import re
import string
import random
import logging
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack, csr_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
random.seed(42)

# ==================== 配置 ====================

class Config:
    """训练配置"""
    
    # 目标设置
    TARGET_AUC = 0.85  # 真实数据上的合理目标
    MAX_OPTIMIZATION_ROUNDS = 5
    
    # 模型设置
    CV_FOLDS = 5
    RANDOM_STATE = 42
    
    # 数据路径
    TRAIN_PATH = './train.csv'
    TEST_PATH = './test.csv'
    OUTPUT_PATH = './'
    
    # 标签列
    LABEL_COL = 'rule_violation'

# ==================== 日志设置 ====================

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('real_competition_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==================== 文本预处理 ====================

class RealDataPreprocessor:
    """真实数据预处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text):
        """清理文本"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 转换为小写
        text = text.lower().strip()
        
        return text
    
    def extract_features(self, df):
        """提取特征"""
        self.logger.info("🔧 开始特征工程...")
        
        # 1. 文本特征
        body_texts = df['body'].apply(self.clean_text)
        rule_texts = df['rule'].apply(self.clean_text)
        
        # 2. TF-IDF特征
        body_tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        ).fit_transform(body_texts)
        
        rule_tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        ).fit_transform(rule_texts)
        
        # 3. 统计特征
        body_length = df['body'].str.len().values.reshape(-1, 1)
        rule_length = df['rule'].str.len().values.reshape(-1, 1)
        word_count = df['body'].str.split().str.len().values.reshape(-1, 1)
        
        # 4. 社区特征（One-hot编码）
        subreddit_encoder = LabelEncoder()
        subreddit_encoded = subreddit_encoder.fit_transform(df['subreddit'].fillna('unknown')).reshape(-1, 1)
        
        # 5. 示例特征
        pos_example_1_len = df['positive_example_1'].str.len().fillna(0).values.reshape(-1, 1)
        pos_example_2_len = df['positive_example_2'].str.len().fillna(0).values.reshape(-1, 1)
        neg_example_1_len = df['negative_example_1'].str.len().fillna(0).values.reshape(-1, 1)
        neg_example_2_len = df['negative_example_2'].str.len().fillna(0).values.reshape(-1, 1)
        
        # 6. 文本相似度特征（简化版）
        body_rule_similarity = np.array([
            len(set(body.split()) & set(rule.split())) / max(len(set(body.split()) | set(rule.split())), 1)
            for body, rule in zip(body_texts, rule_texts)
        ]).reshape(-1, 1)
        
        # 合并所有特征
        statistical_features = np.hstack([
            body_length, rule_length, word_count,
            subreddit_encoded,
            pos_example_1_len, pos_example_2_len,
            neg_example_1_len, neg_example_2_len,
            body_rule_similarity
        ])
        
        # 标准化统计特征
        scaler = StandardScaler()
        statistical_features = scaler.fit_transform(statistical_features)
        
        # 合并TF-IDF和统计特征
        final_features = hstack([body_tfidf, rule_tfidf, statistical_features])
        
        # 转换为CSR格式以便切片
        final_features = final_features.tocsr()
        
        self.logger.info(f"🎉 特征工程完成，最终维度: {final_features.shape}")
        
        return final_features

# ==================== 模型训练器 ====================

class RealCompetitionTrainer:
    """真实比赛训练器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.preprocessor = RealDataPreprocessor()
        
        # 定义模型
        self.models = {
            'LogisticRegression_L1': LogisticRegression(
                C=1.0, penalty='l1', solver='liblinear', random_state=42
            ),
            'LogisticRegression_L2': LogisticRegression(
                C=1.0, penalty='l2', random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'MultinomialNB': MultinomialNB(alpha=0.1)
        }
        
        # 模型权重
        self.model_weights = {
            'LogisticRegression_L1': 0.2,
            'LogisticRegression_L2': 0.2,
            'RandomForest': 0.25,
            'GradientBoosting': 0.25,
            'MultinomialNB': 0.1
        }
    
    def load_data(self):
        """加载数据"""
        self.logger.info("📂 加载真实比赛数据...")
        
        train_df = pd.read_csv(self.config.TRAIN_PATH)
        test_df = pd.read_csv(self.config.TEST_PATH)
        
        self.logger.info(f"✅ 训练数据: {train_df.shape}")
        self.logger.info(f"✅ 测试数据: {test_df.shape}")
        self.logger.info(f"📊 标签分布: {train_df[self.config.LABEL_COL].value_counts().to_dict()}")
        
        return train_df, test_df
    
    def prepare_features(self, train_df, test_df):
        """准备特征"""
        self.logger.info("🔧 准备特征...")
        
        # 合并数据以保持特征一致性
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # 提取特征
        features = self.preprocessor.extract_features(combined_df)
        
        # 分离训练和测试特征
        train_features = features[:len(train_df), :]
        test_features = features[len(train_df):, :]
        
        return train_features, test_features, train_df[self.config.LABEL_COL]
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        model_results = {}
        predictions = []
        
        for name, model in self.models.items():
            self.logger.info(f"  🚀 训练 {name}...")
            
            try:
                # 朴素贝叶斯需要非负特征
                if name == 'MultinomialNB':
                    X_train_nb = np.abs(X_train.toarray() if hasattr(X_train, 'toarray') else X_train)
                    X_val_nb = np.abs(X_val.toarray() if hasattr(X_val, 'toarray') else X_val)
                    model.fit(X_train_nb, y_train)
                    pred = model.predict_proba(X_val_nb)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    pred = model.predict_proba(X_val)[:, 1]
                
                auc = roc_auc_score(y_val, pred)
                model_results[name] = auc
                predictions.append(pred)
                
                self.logger.info(f"    ✅ {name}: {auc:.4f}")
                
            except Exception as e:
                self.logger.error(f"    ❌ {name} 训练失败: {e}")
                model_results[name] = 0.5
                predictions.append(np.random.random(len(y_val)))
        
        return model_results, predictions
    
    def ensemble_predict(self, model_results, predictions, y_val):
        """集成预测"""
        # 加权平均
        weighted_pred = np.zeros(len(y_val))
        total_weight = 0
        
        for i, (name, auc) in enumerate(model_results.items()):
            weight = self.model_weights[name] * auc  # 根据AUC调整权重
            weighted_pred += predictions[i] * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_pred /= total_weight
        
        ensemble_auc = roc_auc_score(y_val, weighted_pred)
        
        return ensemble_auc, weighted_pred
    
    def cross_validation(self, X, y):
        """交叉验证"""
        self.logger.info("🔄 开始交叉验证...")
        
        skf = StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            self.logger.info(f"  📊 折 {fold}/{self.config.CV_FOLDS}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练模型
            model_results, predictions = self.train_models(X_train, y_train, X_val, y_val)
            
            # 集成预测
            ensemble_auc, ensemble_pred = self.ensemble_predict(model_results, predictions, y_val)
            
            fold_results.append({
                'fold': fold,
                'ensemble_auc': ensemble_auc,
                'model_results': model_results
            })
            
            self.logger.info(f"    🎯 集成AUC: {ensemble_auc:.4f}")
        
        # 计算平均结果
        avg_auc = np.mean([r['ensemble_auc'] for r in fold_results])
        std_auc = np.std([r['ensemble_auc'] for r in fold_results])
        
        self.logger.info(f"📈 交叉验证结果: {avg_auc:.4f} ± {std_auc:.4f}")
        
        return avg_auc, fold_results
    
    def train_final_model(self, X_train, y_train, X_test):
        """训练最终模型"""
        self.logger.info("🏆 训练最终模型...")
        
        # 训练所有模型
        final_models = {}
        test_predictions = []
        
        for name, model in self.models.items():
            self.logger.info(f"  🚀 训练最终 {name}...")
            
            try:
                if name == 'MultinomialNB':
                    X_train_nb = np.abs(X_train.toarray() if hasattr(X_train, 'toarray') else X_train)
                    X_test_nb = np.abs(X_test.toarray() if hasattr(X_test, 'toarray') else X_test)
                    model.fit(X_train_nb, y_train)
                    pred = model.predict_proba(X_test_nb)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    pred = model.predict_proba(X_test)[:, 1]
                
                final_models[name] = model
                test_predictions.append(pred)
                
            except Exception as e:
                self.logger.error(f"    ❌ {name} 训练失败: {e}")
                test_predictions.append(np.random.random(X_test.shape[0]))
        
        # 集成预测
        final_pred = np.zeros(X_test.shape[0])
        total_weight = 0
        
        for i, name in enumerate(self.models.keys()):
            weight = self.model_weights[name]
            final_pred += test_predictions[i] * weight
            total_weight += weight
        
        if total_weight > 0:
            final_pred /= total_weight
        
        return final_pred
    
    def generate_submission(self, predictions, test_df):
        """生成提交文件"""
        self.logger.info("📝 生成提交文件...")
        
        submission_df = pd.DataFrame({
            'row_id': test_df['row_id'],
            'rule_violation': predictions
        })
        
        submission_path = f"{self.config.OUTPUT_PATH}submission_real_competition.csv"
        submission_df.to_csv(submission_path, index=False)
        
        self.logger.info(f"✅ 提交文件已保存: {submission_path}")
        self.logger.info(f"📊 预测统计:")
        self.logger.info(f"  均值: {predictions.mean():.4f}")
        self.logger.info(f"  标准差: {predictions.std():.4f}")
        self.logger.info(f"  范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        return submission_path
    
    def train(self):
        """主训练流程"""
        start_time = time.time()
        
        self.logger.info("🚀 开始真实比赛训练流程...")
        
        # 1. 加载数据
        train_df, test_df = self.load_data()
        
        # 2. 准备特征
        X_train, X_test, y_train = self.prepare_features(train_df, test_df)
        
        # 3. 交叉验证
        cv_auc, fold_results = self.cross_validation(X_train, y_train)
        
        # 4. 检查是否达到目标
        if cv_auc >= self.config.TARGET_AUC:
            self.logger.info(f"🎉 目标达成！AUC {cv_auc:.4f} >= {self.config.TARGET_AUC}")
        else:
            self.logger.info(f"⚠️ 未达到目标，当前AUC {cv_auc:.4f} < {self.config.TARGET_AUC}")
        
        # 5. 训练最终模型
        final_predictions = self.train_final_model(X_train, y_train, X_test)
        
        # 6. 生成提交文件
        submission_path = self.generate_submission(final_predictions, test_df)
        
        # 7. 生成报告
        end_time = time.time()
        self.generate_report(cv_auc, fold_results, start_time, end_time, submission_path)
        
        return cv_auc, submission_path
    
    def generate_report(self, final_auc, fold_results, start_time, end_time, submission_path):
        """生成训练报告"""
        training_time = end_time - start_time
        
        self.logger.info("\n" + "="*80)
        self.logger.info("🏆 真实比赛训练完成报告")
        self.logger.info("="*80)
        self.logger.info(f"⏱️  总训练时间: {training_time/60:.1f} 分钟")
        self.logger.info(f"🎯 最终AUC: {final_auc:.4f}")
        self.logger.info(f"📊 目标AUC: {self.config.TARGET_AUC}")
        self.logger.info(f"✅ 目标达成: {'是' if final_auc >= self.config.TARGET_AUC else '否'}")
        self.logger.info(f"📁 提交文件: {submission_path}")
        
        self.logger.info("\n📈 各折详细结果:")
        for result in fold_results:
            self.logger.info(f"  折 {result['fold']}: {result['ensemble_auc']:.4f}")
        
        self.logger.info("\n🎉🎉🎉 训练完成！🎉🎉🎉")

# ==================== 主函数 ====================

def main():
    """主函数"""
    print("="*80)
    print("🚀 Jigsaw Agile Community Rules - 真实比赛训练脚本")
    print("🎯 目标：在真实Kaggle比赛数据上训练模型")
    print("="*80)
    
    # 设置日志
    logger = setup_logging()
    
    # 创建配置
    config = Config()
    
    # 记录环境信息
    logger.info(f"🐍 Python版本: {pd.__version__}")
    logger.info(f"📦 Pandas版本: {pd.__version__}")
    logger.info(f"🎯 目标AUC: {config.TARGET_AUC}")
    logger.info(f"🔄 交叉验证折数: {config.CV_FOLDS}")
    logger.info(f"📂 训练数据路径: {config.TRAIN_PATH}")
    logger.info(f"📂 测试数据路径: {config.TEST_PATH}")
    logger.info(f"🚀 启动训练流程...")
    
    try:
        # 创建训练器
        trainer = RealCompetitionTrainer(config)
        
        # 开始训练
        final_auc, submission_path = trainer.train()
        
        print(f"\n🎉🎉🎉 训练成功！最终AUC: {final_auc:.4f} 🎉🎉🎉")
        print(f"📁 提交文件已生成: {submission_path}")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        raise

if __name__ == "__main__":
    main() 