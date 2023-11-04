import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import argparse
from datetime import datetime
import os
import data as dt

DataName = "NGSIM_US101"
# physical_model = "IDM"
# physical_model = "FVD"
physical_model = "Newell"


def predict_function(num_samples, seed, feature_num):
    backward = 50
    forward = 50

    # 准备数据
    _, _, test_x, _, _, test_rows, a_residual_IDM_min, a_residual_IDM_max, test_chain_ids = dt.load_data(num_samples, seed)
    test_x = test_x.reshape(test_x.shape[0], backward, feature_num)

    # 加载模型
    model = load_model(f"./model/{DataName}.h5")

    # 在测试集上进行预测
    A_residual_hat = model.predict(test_x)

    # 反归一化
    A_residual_hat = A_residual_hat * (a_residual_IDM_max - a_residual_IDM_min) + a_residual_IDM_min

    # 找到原始数据作为对比
    df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/{DataName}_{physical_model}_results.csv")
    indices = []
    for chain_id in test_chain_ids:
        chain_df = df[df['chain_id'] == chain_id]
        indices.extend(chain_df.index[-forward:])
    # 使用这些索引从A_IDM中提取数据
    A_IDM_array = df[f'a_{physical_model}'].iloc[indices].to_numpy()
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
        V_PERL[:, i] = V_PERL[:, i - 1] + A_PERL[:, i] * 0.1

    Y_PERL = np.zeros_like(Y)
    Y_PERL[:, 0:2] = Y[:, 0:2]
    for i in range(2, forward):
        Y_PERL[:, i] = Y_PERL[:, i - 1] + V_PERL[:, i] * 0.1 + A_PERL[:, i] * 0.005


    # 保存结果
    pd.DataFrame(test_chain_ids).to_csv(f'./results_{DataName}/test_chain_ids.csv', index=False)
    pd.DataFrame(A_PERL).to_csv(f'./results_{DataName}/A.csv', index=False)
    pd.DataFrame(V_PERL).to_csv(f'./results_{DataName}/V.csv', index=False)
    pd.DataFrame(Y_PERL).to_csv(f'./results_{DataName}/Y.csv', index=False)


    # 计算MSE，保存
    a_mse = mean_squared_error(A, A_PERL)
    a_mse_first = mean_squared_error(A[:, 0], A_PERL[:, 0])
    v_mse = mean_squared_error(V, V_PERL)
    v_mse_first = mean_squared_error(V[:, 2], V_PERL[:, 2])
    y_mse = mean_squared_error(Y, Y_PERL)
    y_mse_first = mean_squared_error(Y[:, 2], Y_PERL[:, 2])
    with open(f"./results_{DataName}/predict_MSE_results.txt", 'a') as f:
        # now = datetime.now()
        # current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        # f.write(f'{current_time}\n')
        # f.write(f'num_samples ={num_samples}\n')
        f.write(f'{a_mse:.4f},{a_mse_first:.4f},{v_mse:.4f},{v_mse_first:.4f}\n')
        # f.write(f'MSE when predict multi-step a: {a_mse:.5f}\n')
        # f.write(f'MSE when predict first a: {a_mse_first:.5f}\n')
        # f.write(f'MSE when predict multi-step v: {v_mse:.5f}\n')
        # f.write(f'MSE when predict first v: {v_mse_first:.5f}\n\n')
        # f.write(f'MSE when predict multi-step y: {y_mse:.5f}\n')
        # f.write(f'MSE when predict first y: {y_mse_first:.5f}\n\n')


    # 绘制预测结果和真实值的图形
    # os.makedirs(f'./results_{DataName}/plots', exist_ok=True)
    # for i in range(min(60, n_samples)):
    #     plt.figure(figsize=(6, 3))
    #     x = range(len(A_PERL[i]))
    #     plt.plot(x, A[i, :], color='black', markersize=0.5, label='Real-world')
    #     plt.plot(x, A_IDM[i, :], color='g', markersize=0.2, label='physics')
    #     plt.plot(x, A_PERL[i, :], color='r', markersize=0.5, label='PERL')
    #     plt.xlabel('Time Index (0.1 s)')
    #     plt.ylabel('Acceleration $(m/s^2)$')
    #     plt.ylim(-4, 4)
    #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #     plt.legend()
    #     plt.savefig(f'./results_{DataName}/plots/PERL_result{i}.png')
    #     plt.close()


# if __name__ == '__main__':
#     predict_function(30000,3)