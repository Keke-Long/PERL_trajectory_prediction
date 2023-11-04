from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Embedding, LSTM, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
import data as dt
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, schedules

import sys
sys.path.append('/home/ubuntu/Documents/PERL/models/')
from Data_driven_LSTM import data as dt
from predict import predict_function

def build_mlp_model(input_shape, output_units):
    model = Sequential([
        Dense(20, activation='relu', input_shape=input_shape),
        Dense(10, activation='relu'),
        Dense(output_units)
    ])

    return model


if __name__ == '__main__':
    DataName = "NGSIM_US101"

    # 准备数据
    train_x, val_x, _, train_y, val_y, _, _, _, _ = dt.load_data()

    # 加载模型
    model = build_mlp_model((train_x.shape[1],), train_y.shape[1])

    # 使用学习率衰减策略
    #lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.95)
    lr_schedule = 0.001
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')

    # 添加早停策略
    early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00001, verbose=2)

    # 保存收敛过程
    history = model.fit(train_x, train_y,
                        validation_data=(val_x, val_y),
                        epochs=1000, batch_size=64, verbose=2,
                        callbacks=[early_stopping])
    combined_loss_history = np.column_stack((history.history['loss'], history.history['val_loss']))
    os.makedirs(f'./results_{DataName}', exist_ok=True)
    np.savetxt(f"./results_{DataName}/convergence_rate.csv", combined_loss_history, delimiter=",", header="train_loss,val_loss")

    # 保存模型
    model.save(f"./model/{DataName}.h5")

    # 保存模型结构图
    #plot_model(model, to_file='./results/model_structure PERL IDM_MLP.png', show_shapes=True, show_layer_names=True)

    predict_function()