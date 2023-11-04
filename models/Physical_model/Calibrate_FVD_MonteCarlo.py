import pandas as pd
import random
from FVD import FVD
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def monte_carlo_optimization(df, num_iterations):
    best_rmse = 100000
    best_arg = None

    # 使用tqdm显示进度条
    with tqdm(total=num_iterations, desc='Iterations', postfix={'Best MSE': float('inf')}) as pbar:
        for _ in range(num_iterations):
            alpha = random.uniform(0.1, 0.2)
            lamda = random.uniform(0.5, 0.9)
            v_0 = random.uniform(15, 30)
            b = random.uniform(5, 12)
            beta = random.uniform(2, 8)
            arg = (round(alpha, 3), round(lamda, 3), round(v_0, 3), round(b, 3), round(beta, 3))

            df['a_hat'] = df.apply(lambda row: FVD(arg, row['v'], row['v'] - row['v-1'], row['y-1']-row['y']),axis=1)
            df['a_error'] = df['a_hat'] - df['a']

            mse = mean_squared_error(df['a'], df['a_hat'])
            rmse = np.sqrt(mse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_arg = arg

            # 更新最小MSE的值
            pbar.set_postfix_str({'Best RMSE': round(best_rmse, 3), 'best_arg': best_arg})
            pbar.update(1)

    # plt.hist(df['A_error'], bins=20, color='blue', alpha=0.5)
    # plt.title('A_error Distribution')
    return best_arg, best_rmse


# 加载原始数据
import sys
sys.path.append('/home/ubuntu/Documents/PERL/models')  # 将 load_data.py 所在的目录添加到搜索路径
# import load_data_fun
# df = load_data_fun.load_data()

# Load cleaned data 即已经删除了不合理a_IDM_2的数据，防止异常值对标定结果的影响
df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_FVD_results_origin.csv")


# 筛选
df = df[df['Preceding'] != 0]
df = df.dropna(subset=['v', 'v-1', 'Space_Headway'])
print('Before filtering len(df)=', len(df))
df = df[(df['Space_Headway'] > 4) & (df['Space_Headway'] < 150)]
print('After filtering  len(df)=', len(df))


# 随机选取一部分数据进行标定，
df = df.sample(n=1000*50, random_state=1)  # Fixing the random state for reproducibility
print('After sampling len(df_sampled)=', len(df))

# 标定
best_arg, best_rmse = monte_carlo_optimization(df, num_iterations = 1000)

# 结果保存
# {'Best RMSE': 5.586, 'best_arg': (0.676, 0.761, 25.11, 22.423, 2.299)}]
# {'Best RMSE': 2.952, 'best_arg': (0.19, 0.989, 25.403, 6.247, 7.295)}]
# {'Best RMSE': 2.243, 'best_arg': (0.159, 0.541, 25.183, 9.147, 7.256)}]
# {'Best RMSE': 2.121, 'best_arg': (0.15, 0.528, 17.131, 6.87, 2.779)}]
# {'Best RMSE': 1.805, 'best_arg': (0.107, 0.537, 22.971, 9.411, 5.013)}]

# 清洗数据之后：
# {'Best RMSE': 1.465, 'best_arg': (0.11, 0.537, 17.09, 11.929, 2.067)}]