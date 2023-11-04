import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse
import os
from datetime import datetime
import tensorflow as tf

import sys
sys.path.append('/home/ubuntu/Documents/PERL/models/')
from Data_driven_LSTM import data as dt


# DataName = "NGSIM_I80"
DataName = "NGSIM_US101"


def plot_and_save_prediction(A_real, A_MLP, sample_id):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(A_real)), A_real, color='b', markersize=1, label='Real-world')
    plt.plot(np.arange(len(A_real)), A_MLP, color='r', markersize=1, label='LSTM')
    plt.xlabel('Index')
    plt.ylabel('Acceleration error $(m/s^2)$')
    plt.ylim(-4, 4)
    plt.title(f'Sample ID: {sample_id}')
    plt.legend()
    plt.savefig(f'./results_{DataName}/plots/predict_result_{sample_id}.png')
    plt.close()


def predict_function():
    forward = 10
    backward = 50
    os.makedirs(f'./results_{DataName}', exist_ok=True)
    
    _,_, test_x, _,_, test_y_real, A_min, A_max, test_chain_ids = dt.load_data()
    model = load_model(f"./model/{DataName}.h5")
    test_y_predict = model.predict(test_x)

    A_real = test_y_real.tolist()
    A_MLP  = test_y_predict.tolist()

    # 反归一化
    A_real = np.array(A_real) * (A_max - A_min) + A_min
    A_MLP  = np.array(A_MLP) * (A_max - A_min) + A_min

    # 找到原始数据作为对比
    df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/{DataName}_IDM_results.csv")
    indices = []
    for chain_id in test_chain_ids:
        chain_df = df[df['chain_id'] == chain_id]
        indices.extend(chain_df.index[-forward:])
    # 使用这些索引从A_IDM中提取数据
    A_array = df['a'].iloc[indices].to_numpy()
    n_samples = len(A_array) // forward
    A = A_array.reshape(n_samples, forward)

    V_array = df['v'].iloc[indices].to_numpy()
    V = V_array.reshape(n_samples, forward)

    Y_array = df['y'].iloc[indices].to_numpy()
    Y = Y_array.reshape(n_samples, forward)

    V_MLP = np.zeros_like(V)
    V_MLP[:, 0] = V[:, 0]
    for i in range(1, forward):
        V_MLP[:, i] = V[:, i - 1] + A_MLP[:, i - 1] * 0.1

    Y_MLP = np.zeros_like(Y)
    Y_MLP[:, 0:2] = Y[:, 0:2]
    for i in range(2, forward):
        Y_MLP[:, i] = Y[:, i - 1] + V_MLP[:, i - 1] * 0.1 + A_MLP[:, i - 1] * 0.005

    # 保存结果
    pd.DataFrame(test_chain_ids).to_csv(f'./results_{DataName}/test_chain_ids.csv', index=False)
    pd.DataFrame(A_MLP).to_csv(f'./results_{DataName}/A.csv', index=False)
    pd.DataFrame(V_MLP).to_csv(f'./results_{DataName}/V.csv', index=False)
    pd.DataFrame(Y_MLP).to_csv(f'./results_{DataName}/Y.csv', index=False)

    # 计算MSE，保存
    a_mse = mean_squared_error(A, A_MLP)
    a_mse_first = mean_squared_error(A[:, 0], A_MLP[:, 0])
    v_mse = mean_squared_error(V, V_MLP)
    v_mse_first = mean_squared_error(V[:, 1], V_MLP[:, 1])
    y_mse = mean_squared_error(Y, Y_MLP)
    y_mse_first = mean_squared_error(Y[:, 2], Y_MLP[:, 2])
    with open(f"./results_{DataName}/predict_MSE_results.txt", 'a') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'{current_time}\n')
        f.write(f'MSE when predict multi-step a: {a_mse:.5f}\n')
        f.write(f'MSE when predict first a: {a_mse_first:.5f}\n')
        f.write(f'MSE when predict multi-step v: {v_mse:.5f}\n')
        f.write(f'MSE when predict first v: {v_mse_first:.5f}\n')
        f.write(f'MSE when predict multi-step y: {y_mse:.5f}\n')
        f.write(f'MSE when predict first y: {y_mse_first:.5f}\n\n')

    # os.makedirs(f'./results_{DataName}/plots', exist_ok=True)
    # for i in range(len(A_real)):
    #     plot_and_save_prediction(A_real[i], A_MLP[i], i)


if __name__ == '__main__':
    predict_function()