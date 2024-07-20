import pandas as pd
import numpy as np
from jqdatasdk import auth, get_price
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据获取
auth('15995118116', 'Boeing747@142857')

# 获取多只股票数据
def get_stock_data(stock_codes, start_date, end_date):
    stock_data = {}
    for stock_code in stock_codes:
        df = get_price(stock_code, start_date=start_date, end_date=end_date, frequency='daily', fields=['open', 'close', 'high', 'low', 'volume'])
        df['stock_code'] = stock_code
        stock_data[stock_code] = df
    return stock_data

#stock_codes = ['000001.XSHE', '000002.XSHE', '000009.XSHE','000012.XSHE']  # 示例股票代码
#stock_data = get_stock_data(stock_codes, '2023-03-17', '2024-03-23')

# 2. 数据预处理
def preprocess_stock_data(stock_data):
    for code, df in stock_data.items():
        df['next_open'] = df['open'].shift(-1)
        df['prev_close'] = df['close'].shift(1)
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(window=5).std() * np.sqrt(5)
        df.dropna(inplace=True)
        df['D'] = (df['open'] > df['prev_close']).astype(int)
        stock_data[code] = df
    return stock_data

#stock_data = preprocess_stock_data(stock_data)

# 保存每只股票的数据为CSV文件
#for code, df in stock_data.items():
    #df.to_csv(f'{code}_data.csv',mode='w')

# 3. 数据处理与数据划分
def preprocess_and_split(stock_data, train_ratio):
    X_train={}
    X_test={}
    y_train_reg={}
    y_test_reg={}
    y_train_class={}
    y_test_class={}

    for code, df in stock_data.items():
        features = ['open', 'high', 'low', 'close', 'volume', 'volatility']
        X = df[features]
        y_REG = df['next_open']
        y_CLASS = df['D']

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 按时间顺序划分数据集
        train_size = int(len(X_scaled) * train_ratio)
        X_train[code], X_test[code] = X_scaled[:train_size], X_scaled[train_size:]
        y_train_reg[code], y_test_reg[code] = y_REG[:train_size], y_REG[train_size:]
        y_train_class[code], y_test_class[code] = y_CLASS[:train_size], y_CLASS[train_size:]
        

    return X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class

#X_train, X_test, y_train, y_test = preprocess_and_split(stock_data, 0.8)


# 数据可视化
def plot_stock_data(stock_data, stock_codes):
    num_stocks = len(stock_codes)
    fig, axes = plt.subplots(nrows=(num_stocks + 1) // 2, ncols=2, figsize=(14, 7 * ((num_stocks + 1) // 2)))
    
    for i, stock_code in enumerate(stock_codes):
        ax = axes[i // 2, i % 2]
        df = stock_data[stock_code]
        ax.plot(df.index, df['open'], label='Open Price')
        ax.plot(df.index, df['close'], label='Close Price')
        ax.set_title(f'Stock Prices for {stock_code}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
    
    plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)  # 增加子图间的间距
    plt.savefig('stock_prices.jpg')  # 保存为 JPG 图片
    plt.show()

def plot_correlation_matrix(stock_data, stock_codes):
    num_stocks = len(stock_codes)
    fig, axes = plt.subplots(nrows=(num_stocks + 1) // 2, ncols=2, figsize=(10, 8 * ((num_stocks + 1) // 2)))
    
    for i, stock_code in enumerate(stock_codes):
        ax = axes[i // 2, i % 2]
        df = stock_data[stock_code]
        corr = df[['open', 'high', 'low', 'close', 'volume', 'volatility']].corr()
        sns.heatmap(corr.iloc[:, ::-1], annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)  # Reverse the order of columns
        ax.set_title(f'Correlation Matrix for {stock_code}')
    
    plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)  # 增加子图间的间距
    plt.savefig('correlation_matrix.jpg')  # 保存为 JPG 图片
    plt.show()

# 示例：绘制股票数据和相关性热图
#plot_stock_data(stock_data, stock_codes)
#plot_correlation_matrix(stock_data, stock_codes)


