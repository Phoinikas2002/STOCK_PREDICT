import pandas as pd
import numpy as np
import stock_data_process as sdp
from jqdatasdk import auth, get_price
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib as mpl

from keras import Sequential
import keras

# 1. 认证并获取数据
auth('15995118116', 'Boeing747@142857')

stock_codes = ['000001.XSHE', '000002.XSHE', '000009.XSHE','000012.XSHE']  # 示例股票代码
stock_data = sdp.preprocess_stock_data(sdp.get_stock_data(stock_codes, '2023-03-17', '2024-03-23'))


# 2. 模型训练与评估
# 按时间顺序划分数据集
# 80%数据用于训练，20%数据用于测试
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Reshape input for LSTM [samples, time steps, features]
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # 训练LSTM模型进行回归预测
    model_lstm = Sequential()
    model_lstm.add(keras.layers.LSTM(units=250, return_sequences=True, input_shape=(1, X_train.shape[1])))
    model_lstm.add(keras.layers.LSTM(units=250))
    model_lstm.add(keras.layers.Dense(1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

    history = keras.callbacks.History()
    model_lstm.fit(X_train_lstm, y_train, epochs=250, batch_size=32, verbose=0, callbacks=[history])

    # 回归模型评估
    y_pred_reg = model_lstm.predict(X_test_lstm)
    y_pred_reg = np.squeeze(y_pred_reg)
    mae = mean_absolute_error(y_test, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
    return y_pred_reg, mae, rmse, history

X_train, X_test, y_train_reg, y_test_reg, *rest= sdp.preprocess_and_split(stock_data, 0.8)

# 3. 训练模型并输出评估指标
y_pred_reg = {}
mae = {}
rmse = {}
histories = {}

for code in stock_codes:
    y_pred_reg[code], mae[code], rmse[code],histories[code] = train_and_evaluate(X_train[code], X_test[code], y_train_reg[code], y_test_reg[code])
    print(f'{code} MAE: {mae[code]:.2f}')
    print(f'{code} RMSE: {rmse[code]:.2f}')


# 4. 绘图

# 绘制训练过程中loss值随epoch变化的曲线
def plot_training_loss(histories, stock_codes):
    num_stocks = len(stock_codes)
    num_plots = (num_stocks + 3) // 4

    histories_list = [histories[code] for code in stock_codes]

    for i in range(num_plots):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
        axes = axes.flatten()

        for j in range(4):
            index = i * 4 + j
            if index >= num_stocks:
                break

            stock_code = stock_codes[index]
            ax = axes[j]
            histories = histories_list[index]
            ax.plot(histories.history['loss'], label='Training Loss')
            ax.set_title(f'Training Loss for {stock_code}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()


        for k in range(j + 1, 4):
            fig.delaxes(axes[k])

        plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
        plt.savefig(f'training_loss_group_{i + 1}.jpg')
        plt.show()

plot_training_loss(histories, stock_codes)

# 按四个一组分别绘制每个示例股票的预测数据图表
def plot_prediction(stock_data, stock_codes, y_test_reg, y_pred_reg):
    num_stocks = len(stock_codes)
    num_plots = (num_stocks + 3) // 4  # 计算需要的图表组数，每组最多四个图表

    y_test_reg_list = [y_test_reg[code] for code in stock_codes]
    y_pred_reg_list = [y_pred_reg[code] for code in stock_codes]
    
    for i in range(num_plots):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
        axes = axes.flatten()
        
        for j in range(4):
            index = i * 4 + j
            if index >= num_stocks:
                break
            
            stock_code = stock_codes[index]
            ax = axes[j]
            y_test = y_test_reg_list[index]
            y_pred = y_pred_reg_list[index]

            # 计算MAE和RMSE
            mae_stock = mean_absolute_error(y_test, y_pred)
            rmse_stock = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f'{stock_code} MAE: {mae_stock:.2f}')
            print(f'{stock_code} RMSE: {rmse_stock:.2f}')
            
            # 绘制实际开盘价和预测开盘价
            ax.plot(y_test.values, color='red', label='Real Open Price')
            ax.plot(y_pred, color='blue', label='Predicted Open Price')
            ax.set_title(f'Open Price Prediction for {stock_code}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Open Price')
            ax.legend()
        
        # 隐藏多余的子图
        for k in range(j + 1, 4):
            fig.delaxes(axes[k])
        
        plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)  # 增加子图间的间距
        plt.savefig(f'stock_prediction_group_{i + 1}.jpg')  # 保存为 JPG 图片
        plt.show()

# 示例：绘制股票预测数据
plot_prediction(stock_data, stock_codes, y_test_reg, y_pred_reg)


