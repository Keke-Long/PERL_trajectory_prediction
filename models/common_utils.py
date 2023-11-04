from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization, Reshape, Flatten
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D,  SimpleRNN, Dense, GRU


def build_lstm_model(input_shape, forward):
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(forward))  # 假设输出的时间步数与输入相同
    model.add(Activation("relu"))
    return model


def build_lstm_complex_model(input_shape, forward):
    model = Sequential()
    # First LSTM layer
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    # Second LSTM layer
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    # Output layer
    model.add(Dense(forward))
    model.add(Activation("relu"))
    return model


def build_ann_model(input_shape, forward):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(forward, activation='relu'))
    return model


def build_cnn_model(input_shape, forward):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=8, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(forward))
    return model


def build_rnn_model(input_shape, forward):
    model = Sequential()
    model.add(SimpleRNN(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(64))
    model.add(Dense(forward, activation='relu'))
    return model


def build_GRU_model(input_shape, forward):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(64))
    model.add(Dense(forward, activation='relu'))
    return model
