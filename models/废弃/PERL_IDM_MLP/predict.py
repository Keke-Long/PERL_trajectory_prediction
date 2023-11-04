import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import argparse
import os
from datetime import datetime
import data as dt

import sys
sys.path.append('/home/ubuntu/Documents/PERL/models/')
from PERL_IDM_LSTM import data as dt

# DataName = "NGSIM_I80"
DataName = "NGSIM_US101"


def predict_function():
    forward = 10
    backward = 50
    # 准备数据
    _, _, test_x, _, _, test_rows, a_residual_IDM_min, a_residual_IDM_max, test_chain_ids = dt.load_data(backward=50, forward=10)
    model = load_model(f"./model/{DataName}.h5")

    # 在测试集上进行预测
    A_residual_hat = model.predict(test_x)
    # 反归一化
    A_residual_hat = A_residual_hat * (a_residual_IDM_max - a_residual_IDM_min) + a_residual_IDM_min

    # 找到原始数据作为对比
    df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/{DataName}_IDM_results.csv")
    indices = []
    for chain_id in test_chain_ids:
        chain_df = df[df['chain_id'] == chain_id]
        indices.extend(chain_df.index[-forward:])
    A_IDM_array = df['a_IDM_2'].iloc[indices].to_numpy()
    n_samples = len(A_IDM_array) // forward
    A_IDM = A_IDM_array.reshape(n_samples, forward)

    A_array = df['a'].iloc[indices].to_numpy()
    A = A_array.reshape(n_samples, forward)

    V_array = df['v'].iloc[indices].to_numpy()
    V = V_array.reshape(n_samples, forward)

    Y_array = df['y'].iloc[indices].to_numpy()
    Y = Y_array.reshape(n_samples, forward)

    # 计算A_PERL, V_PERL, Y_PERL
    A_PERL = A_IDM - A_residual_hat

    V_PERL = np.zeros_like(V)
    V_PERL[:, 0] = V[:, 0]
    for i in range(1, forward):
        V_PERL[:, i] = V[:, i-1] + A_PERL[:, i-1]*0.1

    Y_PERL = np.zeros_like(Y)
    Y_PERL[:, 0:2] = Y[:, 0:2]
    for i in range(2, forward):
        Y_PERL[:, i] = Y[:, i-1] + V_PERL[:, i-1]*0.1 + A_PERL[:, i-1]*0.005


    # 保存结果
    pd.DataFrame(test_chain_ids).to_csv(f'./results_{DataName}/test_chain_ids.csv', index=False)
    pd.DataFrame(A_PERL).to_csv(f'./results_{DataName}/A_PERL.csv', index=False)
    pd.DataFrame(V_PERL).to_csv(f'./results_{DataName}/V_PERL.csv', index=False)
    pd.DataFrame(Y_PERL).to_csv(f'./results_{DataName}/Y_PERL.csv', index=False)


    # 计算MSE，保存
    a_mse = mean_squared_error(A, A_PERL)
    a_mse_first = mean_squared_error(A[:, 0], A_PERL[:, 0])
    v_mse = mean_squared_error(V, V_PERL)
    v_mse_first = mean_squared_error(V[:, 1], V_PERL[:, 1])
    y_mse = mean_squared_error(Y, Y_PERL)
    y_mse_first = mean_squared_error(Y[:, 2], Y_PERL[:, 2])
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


    # 绘制预测结果和真实值的图形
    # os.makedirs(f'./results_{DataName}/plots', exist_ok=True)
    # for i in range(min(60, n_samples)):
    #     plt.figure(figsize=(6, 4))
    #     x = range(len(A_PERL[i]))
    #     plt.plot(x, A[i,:],  color='b', markersize=0.5, label='Real-world')
    #     plt.plot(x, A_IDM[i,:], color='g', markersize=0.2, label='IDM')
    #     plt.plot(x, A_PERL[i,:], color='r', markersize=0.5, label='PERL(IDM+NN)')
    #     plt.xlabel('Time Index (0.1 s)')
    #     plt.ylabel('Acceleration $(m/s^2)$')
    #     plt.ylim(-4, 4)
    #     plt.legend()
    #     plt.savefig(f'./results_{DataName}/plots/PINN(IDM+NN)_result{i}.png')
    #     #plt.show()
    #     plt.close()


if __name__ == '__main__':
    predict_function()