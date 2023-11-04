import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime
import tensorflow as tf
import data as dt

DataName = "NGSIM_US101"
# physical_model = "IDM"
# physical_model = "FVD"
physical_model = "Newell"


def predict_function(num_samples, seed, feature_num):
    forward = 50
    backward = 50

    _,_, test_x, _,_, test_y_real, A_min, A_max, test_chain_ids = dt.load_data(num_samples, seed)
    test_x = test_x.reshape(test_x.shape[0], backward, feature_num)

    model = load_model(f"./model/{DataName}.h5")
    test_y_predict = model.predict(test_x)

    A_real = test_y_real.tolist()
    A_LSTM = test_y_predict.tolist()

    # 反归一化
    A_LSTM = np.array(A_LSTM) * (A_max - A_min) + A_min

    # 找到原始数据作为对比
    df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_{physical_model}_results.csv")
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

    V_LSTM = np.zeros_like(V)
    V_LSTM[:, 0] = V[:, 0]
    for i in range(1, forward):
        V_LSTM[:, i] = V_LSTM[:, i - 1] + A_LSTM[:, i] * 0.1

    Y_LSTM = np.zeros_like(Y)
    Y_LSTM[:, 0:2] = Y[:, 0:2]
    for i in range(2, forward):
        Y_LSTM[:, i] = Y_LSTM[:, i - 1] + V_LSTM[:, i] * 0.1 + A_LSTM[:, i] * 0.005


    # 保存结果
    # pd.DataFrame(test_chain_ids).to_csv(f'./results_{DataName}/test_chain_ids.csv', index=False)
    # pd.DataFrame(A_LSTM).to_csv(f'./results_{DataName}/A.csv', index=False)
    # pd.DataFrame(V_LSTM).to_csv(f'./results_{DataName}/V.csv', index=False)
    # pd.DataFrame(Y_LSTM).to_csv(f'./results_{DataName}/Y.csv', index=False)


    # 计算MSE，保存
    a_mse = mean_squared_error(A, A_LSTM)
    a_mse_first = mean_squared_error(A[:, 0], A_LSTM[:, 0])
    v_mse = mean_squared_error(V, V_LSTM)
    v_mse_first = mean_squared_error(V[:, 2], V_LSTM[:, 2])
    y_mse = mean_squared_error(Y, Y_LSTM)
    y_mse_first = mean_squared_error(Y[:, 2], Y_LSTM[:, 2])
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


    # os.makedirs(f'./results_{DataName}/plots', exist_ok=True)
    # for i in range(60):
    #     plt.figure(figsize=(6, 3))
    #     x = range(len(A_real[i]))
    #     plt.plot(x, A[i, :], color='black', markersize=1, label='Real-world')
    #     plt.plot(x, A_LSTM[i, :], color='r', markersize=1, label='Data_driven')
    #     plt.xlabel('Index')
    #     plt.ylabel('Acceleration error $(m/s^2)$')
    #     plt.ylim(-4, 4)
    #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #     plt.legend()
    #     plt.savefig(f'./results_{DataName}/plots/Data_driven_{i}.png')
    #     plt.close()
