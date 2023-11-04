import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization, Reshape, Flatten
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, GlobalAveragePooling1D,  SimpleRNN, Dense, GRU
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
import data as dt
from predict import predict_function
from models.common_utils import *


if __name__ == '__main__':
    # 准备数据
    DataName = "NGSIM_US101"
    backward = 50
    forward = 50
    num_samples = 500
    seed = 22
    feature_num = 15

    # Prepare data
    train_x, val_x, _, train_y, val_y, _, _, _, _ = dt.load_data(num_samples, seed)
    train_x = train_x.reshape(train_x.shape[0], backward, feature_num)
    val_x = val_x.reshape(val_x.shape[0], backward, feature_num)
    train_y = train_y.reshape(train_y.shape[0], forward, 1)
    val_y = val_y.reshape(val_y.shape[0], forward, 1)

    # Load model
    model = build_lstm_complex_model((backward, feature_num), forward)

    # 使用学习率衰减策略
    #lr_schedule = ExponentialDecay(initial_learning_rate=0.0015, decay_steps=200, decay_rate=0.9)
    lr_schedule = 0.001
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')

    # 添加早停策略
    early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0.00001, verbose=2)

    history = model.fit(train_x, train_y,
                        validation_data=(val_x, val_y),
                        epochs=1000, batch_size=64, verbose=2,
                        callbacks=[early_stopping])

    # combined_loss_history = np.column_stack((history.history['loss'], history.history['val_loss']))
    # os.makedirs(f'./results_{DataName}', exist_ok=True)
    # np.savetxt(f"./results_{DataName}/convergence_rate_Newell_{num_samples}.csv", combined_loss_history, delimiter=",", header="train_loss,val_loss")

    # Save model
    model.save(f"./model/{DataName}.h5")
    #plot_model(model, to_file='./results/model_plot IDM LSTM.png', show_shapes=True, show_layer_names=True)

    predict_function(num_samples, seed, feature_num)  # 预测