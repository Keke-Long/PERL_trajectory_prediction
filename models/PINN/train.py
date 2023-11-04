import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Dropout, TimeDistributed, Activation, Flatten, SimpleRNN, GRU
# from keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import data as dt
from custom_layers import IDM_Layer
from datetime import datetime
import os


def combined_loss(w1):
    def loss_fn(y_true, y_pred):
        idm_pred = y_pred[:, :10]
        lstm_pred = y_pred[:, 10:]
        idm_loss = K.mean(K.square(idm_pred - lstm_pred))
        lstm_loss = K.mean(K.square(lstm_pred - y_true))
        return w1 * idm_loss + (1-w1) * lstm_loss

    return loss_fn

def build_PINN_model(train_x_shape, train_y_shape):
    # IDM Layer
    vi_input = Input(shape=(1,), name='vi_input')  # 一个形状为(N, 1)的数组
    delta_v_input = Input(shape=(1,), name='delta_v_input')
    delta_d_input = Input(shape=(1,), name='delta_d_input')
    x_input = Input(shape=(None, 4), name='x_input')  # 一个形状为(N, T, 5)的数组，T是序列的长度。Assuming you've 6 features for LSTM
    # IDM Layer 输出
    idm_output = IDM_Layer(forward_steps=10)([vi_input, delta_v_input, delta_d_input])
    idm_output = Flatten()(idm_output)

    # LSTM Layer
    # lstm_output = LSTM(32, return_sequences=False)(x_input)
    # lstm_output = Dropout(0.1)(lstm_output)
    # lstm_output = Dense(10)(lstm_output)  # 将LSTM输出的每一个时间步通过一个Dense层进行处理
    # lstm_output = Activation("relu")(lstm_output)

    # GRU Layer
    lstm_output = GRU(64, return_sequences=True)(x_input)
    lstm_output = Dropout(0.2)(lstm_output)
    lstm_output = GRU(64)(lstm_output)
    lstm_output = Dense(10)(lstm_output)
    lstm_output = Activation("relu")(lstm_output)

    combined_output = tf.keras.layers.concatenate([idm_output, lstm_output])

    return [vi_input, delta_v_input, delta_d_input, x_input], idm_output, lstm_output, combined_output

def create_training_model(inputs, idm_output, lstm_output, combined_output):
    training_model = Model(inputs=inputs, outputs = combined_output)
    training_model.compile(optimizer=Adam(learning_rate=0.001), loss=combined_loss(w1=0.1))
    return training_model

def create_prediction_model(inputs, idm_output, lstm_output, alpha):
    prediction_model = Model(inputs=inputs, outputs=alpha*lstm_output + (1-alpha)*idm_output )
    return prediction_model


#if __name__ == '__main__':
for k in [2,3,4,5]:
    DataName = "NGSIM_US101"
    backward = 50
    forward = 10
    num_samples = 42000

    # 准备数据
    train_x, val_x, _, train_y, val_y, _, _, _, _ = dt.load_data(num_samples)

    # 将train_x解构为IDM和LSTM的输入部分
    vi_data = np.array([item[0][0] for item in train_x])
    delta_v_data = np.array([item[0][1] for item in train_x])
    delta_d_data = np.array([item[0][2] for item in train_x])
    x_data = np.array([item[1] for item in train_x])
    x_data = x_data.reshape((-1, backward, 4))
    train_y = np.array(train_y)

    vi_val_data = np.array([item[0][0] for item in val_x])
    delta_v_val_data = np.array([item[0][1] for item in val_x])
    delta_d_val_data = np.array([item[0][2] for item in val_x])
    x_val_data = np.array([item[1] for item in val_x])
    x_val_data = x_val_data.reshape((-1, backward, 4))
    val_y = np.array(val_y)


    # 创建模型
    # training_model = create_training_model(x_data.shape, train_y.shape[1])
    inputs, idm_output, lstm_output, combined_output = build_PINN_model(x_data.shape, train_y.shape[1])
    training_model = create_training_model(inputs, idm_output, lstm_output, combined_output)

    # 添加收敛检查
    early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00001, verbose=2)

    # 保存收敛过程
    history = training_model.fit(
        [vi_data, delta_v_data, delta_d_data, x_data], train_y,
        validation_data=([vi_val_data, delta_v_val_data, delta_d_val_data, x_val_data], val_y),
        epochs=1000, batch_size=64, verbose=2, callbacks=[early_stopping]
    )

    combined_loss_history = np.column_stack((history.history['loss'], history.history['val_loss']))
    os.makedirs(f'./results_{DataName}', exist_ok=True)
    np.savetxt(f"./results_{DataName}/r1_{num_samples}_{k}_convergence_rate.csv", combined_loss_history, delimiter=",",
               header="train_loss,val_loss")

    # 保存模型
    training_model.save(f"./model/{DataName}.h5")


    # 输出训练得到的IDM参数
    idm_layer = [layer for layer in training_model.layers if isinstance(layer, IDM_Layer)][0]
    print("vf:", K.get_value(idm_layer.vf))
    print("A:", K.get_value(idm_layer.A))
    print("b:", K.get_value(idm_layer.b))
    print("s0:", K.get_value(idm_layer.s0))
    print("T:", K.get_value(idm_layer.T))

    # with open(f"./results_{DataName}/predict_MSE_results.txt", 'a') as f:
    #     now = datetime.now()
    #     current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    #     f.write(f'{current_time}\n')
    #     f.write(f'IDM parameters trained result vf, A, b, s0, T = '
    #             f'{K.get_value(idm_layer.vf):.4f}'
    #             f'{K.get_value(idm_layer.A):.4f}'
    #             f'{K.get_value(idm_layer.b):.4f}'
    #             f'{K.get_value(idm_layer.s0):.4f}'
    #             f'{K.get_value(idm_layer.T):.4f}\n')
        