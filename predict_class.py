import pandas as pd
import numpy as np
import stock_data_process as sdp
from jqdatasdk import auth
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据获取与预处理
auth('15995118116', 'Boeing747@142857')

stock_codes = ['000001.XSHE', '000002.XSHE', '000009.XSHE', '000012.XSHE']  # 示例股票代码
stock_data = sdp.preprocess_stock_data(sdp.get_stock_data(stock_codes, '2023-03-17', '2024-03-23'))

X_train, X_test, _, _, y_train_class, y_test_class = sdp.preprocess_and_split(stock_data, 0.8)

# 3. 模型训练与评估
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # 使用随机森林进行分类预测
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_class = model_rf.predict(X_test)
    y_pred_proba = model_rf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred_class)
    return y_pred_class, y_pred_proba, accuracy

y_pred_class = {}
y_pred_proba = {}
accuracy = {}

for code in stock_codes:
    y_pred_class[code], y_pred_proba[code], accuracy[code] = train_and_evaluate(X_train[code], X_test[code], y_train_class[code], y_test_class[code])
    print(f'{code} Accuracy: {accuracy[code]:.2f}')

# 4. 绘图 - ROC曲线
plt.figure(figsize=(14, 7))
for code in stock_codes:
    fpr, tpr, _ = roc_curve(y_test_class[code], y_pred_proba[code])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{code} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

