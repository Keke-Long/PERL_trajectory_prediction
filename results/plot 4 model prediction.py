import pandas as pd
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import glob

def load_data(backward=50, forward=10, data_fraction=0.5):
    #df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_IDM_results.csv")
    df = pd.read_csv("/data/NGSIM_haotian/NGSIM_US101_IDM_results PERL.csv")

    df['delta_y'] = df['y-1'] - df['y']

    # Initialize the scaler
    scaler_delta_y = MinMaxScaler()
    scaler_v_1 = MinMaxScaler()
    scaler_v   = MinMaxScaler()
    scaler_a   = MinMaxScaler()

    # Fit the scaler on the entire dataset
    scaler_delta_y.fit(df['delta_y'].values.reshape(-1, 1))
    scaler_v_1.fit(df['v-1'].values.reshape(-1, 1))
    scaler_v.fit(df['v'].values.reshape(-1, 1))
    scaler_a.fit(df['a'].values.reshape(-1, 1))

    a_min = min(df['a'])
    a_max = max(df['a'])
    print(f'a_min = {a_min}, a_max = {a_max}')

    # Initialize the lists to hold the features and targets
    X = []
    Y = []
    V_real = []

    # Initialize the list to hold the chain_ids for each sample
    sample_chain_ids = []

    chain_ids = df['chain_id'].unique()
    for chain_id in chain_ids:
        # Get the subset of the DataFrame for this chain ID
        chain_df = df[df['chain_id'] == chain_id]

        # Normalize the features
        delta_Y_normalized = scaler_delta_y.transform(chain_df['y'].values.reshape(-1, 1))
        V_1_normalized = scaler_v_1.transform(chain_df['v-1'].values.reshape(-1, 1))
        V_normalized   = scaler_v.transform(chain_df['v'].values.reshape(-1, 1))
        A_normalized   = scaler_a.transform(chain_df['a'].values.reshape(-1, 1))

        # Create the feature vectors and targets for each sample in this chain
        for i in range(0, len(chain_df) - backward - forward + 1, backward + forward):
            X_sample = np.concatenate((delta_Y_normalized[i:i + backward, 0],
                                       V_1_normalized[i:i + backward, 0],
                                       V_normalized[i:i + backward, 0],
                                       A_normalized[i:i + backward, 0]), axis=0)
            Y_sample = A_normalized[i + backward:i + backward + forward, 0]
            V_sample = chain_df['v'].values[i + backward:i + backward + forward]
            V_real.append(V_sample)

            X.append(X_sample)
            Y.append(Y_sample)
            sample_chain_ids.append(chain_id)

    # Convert the lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    V_real = np.array(V_real)
    print(f"Original number of samples: {len(X)}")

    # divided into a training + validation set and a test set
    X_temp, X_test, y_temp, y_test, _, V_real, temp_chain_ids, test_chain_ids = train_test_split(X, Y, V_real, sample_chain_ids,
                                                                                      test_size=0.2, random_state=42)
    # divided into training set and validation set
    X_train, X_val, y_train, y_val, train_chain_ids, val_chain_ids = train_test_split(X_temp, y_temp, temp_chain_ids,
                                                                                      test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test, a_min, a_max, test_chain_ids, V_real


DataName = "NGSIM_US101"
forward = 10
backward = 50

_,_, test_x, _,_, test_y_real, A_min, A_max, test_chain_ids, V_real = load_data()

A_real = test_y_real.tolist()
A_real = np.array(A_real) * (A_max - A_min) + A_min # 反归一化

def plot_for_chains(chain_number, A_real):
    model_paths = {
        'Data-driven': '/home/ubuntu/Documents/PERL/models/Data_driven_LSTM/results_NGSIM_US101',
        'PINN': '/home/ubuntu/Documents/PERL/models/PINN_IDM_LSTM/results_NGSIM_US101',
        'PERL': '/home/ubuntu/Documents/PERL/models/PERL_IDM_LSTM/results_NGSIM_US101',
    }

    model_colors = {
        'Data-driven': '#FFA500',
        'PINN': '#9933FF',
        'PERL': '#0073e6',
    }

    # plot acceleration results of all models
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, path in model_paths.items():
        a_model_df = pd.read_csv(os.path.join(path, 'A.csv'))
        a_values = a_model_df.loc[chain_number]
        ax.plot(a_values, label=f"{model}", color=model_colors[model])

    ax.plot(A_real[chain_number], label="Real-world data", linestyle="--", color='black')
    ax.set_xlabel('Predict Time steps')
    ax.set_ylabel("Acceleration $(m/s^2)$")
    plt.ylim(-2, 2)
    plt.legend()
    plt.savefig(f"./NGSIM_US101/a_v_prediction/chain{chain_number}_a.png")
    plt.close(fig)

    # plot speed results of all models
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, path in model_paths.items():
        v_model_df = pd.read_csv(os.path.join(path, 'V.csv'))
        v_values = v_model_df.loc[chain_number]
        ax.plot(v_values, label=f"{model}", color=model_colors[model])

    ax.plot(V_real[chain_number], label="Real-word data", linestyle="--", color='black')
    ax.set_xlabel('Predict Time steps')
    ax.set_ylabel("Speed $(m/s)$")
    plt.ylim( min(v_values)-0.3, min(v_values)+0.7)
    plt.legend()
    plt.savefig(f"./NGSIM_US101/a_v_prediction/chain{chain_number}_v.png")
    plt.close(fig)

# Assuming A_real is a predefined array containing real A values for all chains.
# Plot for the first 10 chains
for chain_number in range(100):
    plot_for_chains(chain_number, A_real)
