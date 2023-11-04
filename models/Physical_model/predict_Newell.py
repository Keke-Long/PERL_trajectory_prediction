import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


w = 4.44445 # m/s


def predict(filepath):
    data = pd.read_csv(filepath)

    # Re-initializing the predicted speeds list to cover the entire dataset
    begin = 20
    predicted_v = [None] * begin  # First several entries will be None
    predicted_a = [None] * begin

    # 遍历未知数据的每一行
    for index in range(begin, len(data)):
        row = data.iloc[index]
        found_speed = False
        # 遍历上游的车辆，从veh_ID-1开始
        for i in range(1, 5):
            # 根据shockwave计算应该使用的上游车辆的时刻
            time_difference = (row[f'Y-{i}']-row['Y0']) / w
            if time_difference < 0:
                # 如果时间差为负数，这意味着上游车辆的位置在主车辆之后，我们应该跳过这辆车
                continue
            # 获取这个时刻的上游车辆的速度
            refer_index = index - int(time_difference)*10
            if refer_index >= 0:
                predicted_v.append(data.iloc[refer_index][f'v-{i}'])
                predicted_a.append(data.iloc[refer_index][f'a-{i}'])
                found_speed = True
                break

        # 如果未找到任何上游车辆的信息
        if not found_speed:
            #print(index, found_speed)
            predicted_v.append(predicted_v[-1])
            predicted_a.append(predicted_a[-1])

    # 计算预测的加速度
    # predicted_a += [0] + [(predicted_v[i] - predicted_v[i-1]) for i in range(101, len(predicted_v))]

    data['v0_Newell'] = predicted_v
    data['a0_Newell'] = predicted_a
    data['a0_residual_Newell'] = data['a0_Newell'] - data['a0']
    data.to_csv(filepath, index=False)


def plot_predicted_value(data):
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['v0'], label='Actual Speed', color='blue')
    plt.plot(data['t'], data['v0_Newell'], label='Predicted Speed', linestyle='--', color='red')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.title('Actual vs. Predicted Speed')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['a0'], label='Actual acceleration', color='blue')
    plt.plot(data['t'], data['a0_Newell'], label='Predicted acceleration', linestyle='--', color='red')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title('Actual vs. Predicted Acceleration')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate(filepath):
    data = pd.read_csv(filepath)
    v_mse = ((data['v0'] - data['v0_Newell'])**2).mean()
    a_mse = ((data['a0'] - data['a0_Newell'])**2).mean()
    return v_mse, a_mse



# # Prediction
# path = "/home/ubuntu/Documents/PERL/data/NGSIM_haotian1/NGSIM_US101/"
# all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
# for file in tqdm(all_files, desc="Processing files"):
#     predict(os.path.join(path, file))


# # Evaluation
# v_mses = []
# a_mses = []
# path = "/home/ubuntu/Documents/PERL/data/NGSIM_haotian1/NGSIM_US101/"
# all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
# for file in tqdm(all_files, desc="Processing files"):
#     v_mse, a_mse = evaluate(os.path.join(path, file))
#     v_mses.append(v_mse)
#     a_mses.append(a_mse)
#
# v_mses_clean = [mse for mse in v_mses if not pd.isna(mse)]
# a_mses_clean = [mse for mse in a_mses if not pd.isna(mse)]
# avg_v_mse = sum(v_mses_clean) / len(v_mses_clean)
# avg_a_mse = sum(a_mses_clean) / len(a_mses_clean)
#
# print(f"Newell_Average Speed MSE: {avg_v_mse}")
# print(f"Newell_Average Acceleration MSE: {avg_a_mse}")



# Concatenate all DataFrames in the list
# path = "/home/ubuntu/Documents/PERL/data/NGSIM_haotian1/NGSIM_US101/"
# all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
# dfs = []
# chain_id = 0
# for file in tqdm(all_files, desc="Processing files"):
#     df = pd.read_csv(os.path.join(path, file), skiprows=range(1, 51), nrows=100)
#     if df.isna().sum().sum() == 0 \
#             and all(df['a0_residual_Newell'] > -3.5) and all(df['a0_residual_Newell'] < 3.5) \
#             and all(df['Y-1']-df['Y0'] < 150) and all(df['Y-1']-df['Y0'] > 4):
#         df['chain_id'] = chain_id
#         dfs.append(df)
#         chain_id += 1
# combined_df = pd.concat(dfs, ignore_index=True)
# print('Max chain_id=',chain_id)
#
# cols_to_round = ['Y0', 'v0', 'a0',
#                  'Y-1', 'v-1', 'a-1',
#                  'Y-2', 'v-2', 'a-2',
#                  'Y-3', 'v-3', 'a-3',
#                  'Y-4', 'v-4', 'a-4',
#                  'v0_Newell', 'a0_Newell', 'a0_residual_Newell']
# combined_df[cols_to_round] = combined_df[cols_to_round].round(4)
# combined_df[cols_to_round] = combined_df[cols_to_round].astype('float32')
# cols_to_convert = ['veh_ID0', 'veh_ID-1', 'veh_ID-2', 'veh_ID-3', 'veh_ID-4']
# for col in cols_to_convert:
#     combined_df[col] = combined_df[col].astype('int32')
#
# column_mapping = {'Y0': 'y',
#                   'v0': 'v',
#                   'a0': 'a',
#                   'Y-1': 'y-1',
#                   'v0_Newell': 'v_Newell',
#                   'a0_Newell': 'a_Newell',
#                   'a0_residual_Newell': 'a_residual_Newell'}
# combined_df.rename(columns=column_mapping, inplace=True)
#
# combined_df.to_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_Newell_results.csv", index=False)




from sklearn.metrics import mean_squared_error

combined_df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_Newell_results.csv")
combined_df = combined_df.reset_index(drop=True)

# # Filter the rows 51-100 for each chain
# subset = combined_df.groupby('chain_id').apply(lambda x: x.iloc[50:100]).reset_index(drop=True)
# mse_a = mean_squared_error(subset['a'], subset['a_Newell'])
# mse_v = mean_squared_error(subset['v'], subset['v_Newell'])
# print(f'MSE for a0 vs a_Newell (rows 51-100): {mse_a}')
# print(f'MSE for v0 vs v_Newell (rows 51-100): {mse_v}')
#
# # Filter the row 51 for each chain
# subset_51 = combined_df.groupby('chain_id').apply(lambda x: x.iloc[50]).reset_index(drop=True)
# mse_a_51 = mean_squared_error(subset_51['a'], subset_51['a_Newell'])
# mse_v_51 = mean_squared_error(subset_51['v'], subset_51['v_Newell'])
# print(f'MSE for a0 vs a_Newell (row 51): {mse_a_51}')
# print(f'MSE for v0 vs v_Newell (row 51): {mse_v_51}')

def compute_v_newell(chain_group):
    # Compute the 51st row's v_Newell
    chain_group.loc[chain_group.index[51], 'v_Newell'] = chain_group.loc[chain_group.index[50], 'v'] + \
                                                         chain_group.loc[chain_group.index[51], 'a_Newell'] * 0.1
    # For rows from 52 onwards, compute v_Newell
    for i in range(52, len(chain_group)):
        chain_group.loc[chain_group.index[i], 'v_Newell'] = chain_group.loc[chain_group.index[i - 1], 'v_Newell'] + \
                                                            chain_group.loc[chain_group.index[i], 'a_Newell'] * 0.1
    return chain_group

combined_df = combined_df.groupby('chain_id').apply(compute_v_newell).reset_index(drop=True)

# Filter the rows 51-100 for each chain
subset = combined_df.groupby('chain_id').apply(lambda x: x.iloc[51:100]).reset_index(drop=True)
mse_a = mean_squared_error(subset['a'], subset['a_Newell'])
mse_v = mean_squared_error(subset['v'], subset['v_Newell'])
print(f'MSE for a0 vs a_Newell (rows 51-100): {mse_a}')
print(f'MSE for v0 vs v_Newell (rows 51-100): {mse_v}')

# Filter the row 51 for each chain
subset_51 = combined_df.groupby('chain_id').apply(lambda x: x.iloc[51]).reset_index(drop=True)
mse_a_51 = mean_squared_error(subset_51['a'], subset_51['a_Newell'])
mse_v_51 = mean_squared_error(subset_51['v'], subset_51['v_Newell'])
print(f'MSE for a0 vs a_Newell (row 51): {mse_a_51}')
print(f'MSE for v0 vs v_Newell (row 51): {mse_v_51}')
