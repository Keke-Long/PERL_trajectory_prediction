import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


backward = 50
forward = 10
t_chain = backward + forward

physical_model = "IDM"
#physical_model = "FVD"

def load_data(num_samples):
    #df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_IDM_results.csv")
    df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_{physical_model}_results PERL.csv")

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

    # Initialize the list to hold the chain_ids for each sample
    sample_chain_ids = []

    chain_ids = df['chain_id'].unique()
    for chain_id in chain_ids:
        # Get the subset of the DataFrame for this chain ID
        chain_df = df[df['chain_id'] == chain_id]

        # Normalize the features
        delta_Y_normalized = scaler_delta_y.transform(chain_df['delta_y'].values.reshape(-1, 1))
        V_1_normalized = scaler_v_1.transform(chain_df['v-1'].values.reshape(-1, 1))
        V_normalized   = scaler_v.transform(chain_df['v'].values.reshape(-1, 1))
        A_normalized   = scaler_a.transform(chain_df['a'].values.reshape(-1, 1))

        # Create the feature vectors and targets for each sample in this chain
        for i in range(0, len(chain_df) - backward - forward + 1, backward + forward):
            LSTM_input = np.concatenate((delta_Y_normalized[i:i + backward, 0],
                                         V_1_normalized[i:i + backward, 0],
                                         V_normalized[i:i + backward, 0],
                                         A_normalized[i:i + backward, 0]), axis=0)
            vi_0 = df['v'][0]
            delta_v_0 = df['v'][0] - df['v-1'][0]
            delta_d_0 = df['y-1'][0] - df['y'][0]
            IDM_input = [vi_0, delta_v_0, delta_d_0]

            X_sample = [IDM_input, LSTM_input.tolist()]
            Y_sample = A_normalized[i + backward:i + backward + forward, 0]

            X.append(X_sample)
            Y.append(Y_sample)
            sample_chain_ids.append(chain_id)

    # Convert the lists to numpy arrays
    # X = np.array(X)
    # Y = np.array(Y)
    print(f"Original number of samples: {len(X)}")


    # 随机选取一定数量的数据
    # If the total number of samples is greater than desired number, then choose a subset
    np.random.seed(2)
    if len(X) > num_samples:
        indices = np.random.choice(len(X), num_samples, replace=False)
        X = [X[i] for i in indices]
        Y = [Y[i] for i in indices]
        sample_chain_ids = [sample_chain_ids[i] for i in indices]
    print("Xsize",len(X))

    # divided into a training + validation set and a test set
    X_temp, X_test, y_temp, y_test, temp_chain_ids, test_chain_ids = train_test_split(X, Y, sample_chain_ids,
                                                                                      test_size=0.2, random_state=42)
    # divided into training set and validation set
    X_train, X_val, y_train, y_val, train_chain_ids, val_chain_ids = train_test_split(X_temp, y_temp, temp_chain_ids,
                                                                                      test_size=0.25, random_state=42)
    a_min = -3.414
    a_max = 3.414
    return X_train, X_val, X_test, y_train, y_val, y_test, a_min, a_max, test_chain_ids