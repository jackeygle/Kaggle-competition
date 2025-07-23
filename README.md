# Jigsaw Agile Community Rules - 入门模型

这个项目为 Kaggle "Jigsaw Agile Community Rules" 比赛提供了两个版本的入门模型：
- **基础版本** (`jigsaw_baseline_model.py`): 简单易懂的 TF-IDF + 逻辑回归模型
- **增强版本** (`jigsaw_enhanced_model.py`): 包含高级功能的完整解决方案

## 🚀 快速开始

### 在 Kaggle Notebook 中使用

1. 将代码复制到 Kaggle Notebook 中
2. 确保数据路径正确 (`../input/jigsaw-agile-community-rules/`)
3. 运行代码即可生成 `submission.csv` 文件

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行基础模型
python jigsaw_baseline_model.py

# 运行增强模型
python jigsaw_enhanced_model.py
```

## 📁 文件说明

### 核心文件

- `jigsaw_baseline_model.py` - 基础入门模型
- `jigsaw_enhanced_model.py` - 增强版模型
- `requirements.txt` - 依赖包列表

### 输出文件

- `submission.csv` - 基础模型的提交文件
- `submission_enhanced.csv` - 增强模型的提交文件
- `precision_recall_curves.png` - PR曲线图（仅增强版）
- `label_distribution.png` - 标签分布图（仅增强版）

## 🔧 基础模型特点

### 技术栈
- **特征提取**: TF-IDF (词频-逆文档频率)
- **模型**: 逻辑回归 (Logistic Regression)
- **多标签处理**: MultiOutputClassifier
- **库**: scikit-learn, pandas, numpy

### 核心功能
1. **文本预处理**
   - 转换为小写
   - 去除HTML标签
   - 去除特殊字符
   - 标准化空格

2. **特征工程**
   - TF-IDF向量化 (1-gram 和 2-gram)
   - 10,000个最高频特征
   - 英文停用词去除

3. **模型训练**
   - 多标签逻辑回归
   - 80/20 训练验证分割
   - AUC评估指标

4. **输出**
   - 标准格式的submission.csv文件
   - 每个标签的预测概率

## ⭐ 增强模型特点

### 额外功能
1. **高级数据清洗**
   - HTML标签去除
   - URL和邮箱去除
   - 标点符号处理
   - NLTK停用词去除
   - 数字去除
   - 短文本过滤

2. **交叉验证**
   - 5折分层交叉验证
   - F1分数和AUC评估
   - 每折详细结果显示

3. **可视化分析**
   - 标签分布图
   - Precision-Recall曲线
   - 平均精度分数 (AP)

4. **增强特征**
   - 1-3 gram TF-IDF
   - 15,000个特征
   - 更严格的特征筛选

## 📊 预期结果

### 数据格式
比赛通常包含以下标签：
- `toxic` - 毒性评论
- `severe_toxic` - 严重毒性
- `obscene` - 猥亵内容
- `threat` - 威胁
- `insult` - 侮辱
- `identity_hate` - 身份仇恨

### 性能指标
- **评估指标**: ROC-AUC (接收者操作特征曲线下面积)
- **基础模型预期**: AUC 0.85-0.90
- **增强模型预期**: AUC 0.87-0.92

## 🛠️ 自定义和优化

### 调整参数
```python
# TF-IDF参数
max_features = 10000  # 特征数量
ngram_range = (1, 2)  # n-gram范围

# 逻辑回归参数
C = 4  # 正则化强度
solver = 'liblinear'  # 求解器
```

### 添加特征
```python
# 文本长度特征
train_df['text_length'] = train_df['comment_text'].str.len()

# 大写字母比例
train_df['caps_ratio'] = train_df['comment_text'].str.count(r'[A-Z]') / train_df['text_length']
```

## 📈 进一步改进建议

1. **特征工程**
   - 字符级n-gram
   - 情感分析特征
   - 语言检测特征

2. **模型集成**
   - 随机森林
   - 梯度提升树
   - 模型融合

3. **深度学习**
   - LSTM/GRU网络
   - BERT预训练模型
   - 注意力机制

## ⚠️ 注意事项

1. **数据路径**: 确保Kaggle环境中数据路径正确
2. **内存限制**: 大特征数量可能导致内存不足
3. **运行时间**: 增强模型需要更长训练时间
4. **依赖安装**: NLTK在首次使用时需要下载数据

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

MIT License - 可自由使用和修改

---

**祝你在 Kaggle 比赛中取得好成绩！** 🏆 