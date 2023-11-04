import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import matplotlib.path as mpath
import matplotlib.patches as mpatches


# 设置全局字体和大小
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18
tick_font_size = 16

DataName = "NGSIM_US101"

def pad_data(data, max_length):
    """用数据的最后一个值填充数据，直到其长度与max_length一致"""
    return np.pad(data, (0, max_length - len(data)), 'edge')


def plot_data_single(d, case1,case2,case3):
    #只画一个案例的一条线，没有别的
    DataName = "NGSIM_US101"

    # 加载convergence rate数据
    Data_driven_data = np.loadtxt(f"../models/Data_driven/results_{DataName}/{d}_{case1}_convergence_rate.csv", delimiter=",")
    PINN_data = np.loadtxt(f"../models/PINN/results_{DataName}/{d}_{case2}_convergence_rate.csv", delimiter=",")
    PERL_data = np.loadtxt(f"../models/PERL/results_{DataName}/{d}_{case3}_convergence_rate.csv", delimiter=",")

    # 在填充数据之前，保存每个数据集的长度
    data_driven_convergence_point = [len(Data_driven_data)-1]
    pinn_convergence_point = [len(PINN_data)-1]
    perl_convergence_point = [len(PERL_data)-1]

    # 使用pad_data函数确保所有数据的长度与max_length一致
    max_length = max(len(Data_driven_data), len(PINN_data), len(PERL_data))
    Data_driven_data = pad_data(Data_driven_data, max_length)
    PINN_data = pad_data(PINN_data, max_length)
    PERL_data = pad_data(PERL_data, max_length)

    fig, ax = plt.subplots(figsize=(4, 3.2))

    # 橘色系
    #ax.plot(Data_driven_data[:, 0], label="NN Train Loss", color="#ff7700", linestyle='-.', linewidth=1)
    ax.plot(Data_driven_data[:, 1], label="NN", color="#FFA500", linestyle='-', linewidth=1)

    # 紫色系
    #ax.plot(PINN_data[:, 0], label="PINN Train Loss", color="#7A00CC", linestyle=':', linewidth=1.3)
    ax.plot(PINN_data[:, 1], label="PINN", color="#9933FF", linestyle=(1,(5,5)), linewidth=1.3)

    # 蓝色系
    #ax.plot(PERL_data[:, 0], label="PERL Train Loss", color="#0059b3", linestyle='-', linewidth=1.9)
    ax.plot(PERL_data[:, 1], label="PERL", color="#0073e6", linestyle=(1,(5,1)), linewidth=1.6)


    # 绘制三角形
    colors = ["#ff7700", "#7A00CC", "#0059b3"]
    delta = 1.5
    plt.scatter(data_driven_convergence_point, Data_driven_data[data_driven_convergence_point][0][1] * delta,
                marker='v', color=colors[0], s=80, label="NN Convergence")
    plt.scatter(pinn_convergence_point, PINN_data[pinn_convergence_point][0][1] * delta,
                marker='v', color=colors[1], s=80, label="PINN Convergence")
    plt.scatter(perl_convergence_point, PERL_data[perl_convergence_point][0][1] * delta,
                marker='v', color=colors[2], s=80, label="PERL Convergence")

    # 其他设置
    plt.xlabel("Epoch")
    plt.xlim(0, max_length)

    ax.set_ylabel("MSE Loss $(m^2/s^4)$")
    ax.set_yscale("log")  # 设置y轴为对数尺度
    plt.ylim(0.0005, 10)

    ax.tick_params(axis='x', labelsize=tick_font_size)
    ax.tick_params(axis='y', labelsize=tick_font_size)

    plt.subplots_adjust(left=0.26, right=0.92, bottom=0.22, top=0.95)

    #plt.legend(loc='upper right', frameon=False, fontsize=12, ncol=2)

    plt.savefig(f'{d} Convergence Rate new.png', dpi=350)
    #plt.show()


def plot_data(d):

    # 存储所有训练损失数据
    data_driven_train_losses = []
    pinn_train_losses = []
    perl_train_losses = []

    # 循环遍历i
    max_train_length = 0  # 存储所有模型中的最大训练损失长度
    for i in range(1, 6):  # 假设有5个文件
        data_driven_file = f"../models/Data_driven/results_{DataName}/{d}_{i}_convergence_rate.csv"
        pinn_file = f"../models/PINN/results_{DataName}/{d}_{i}_convergence_rate.csv"
        perl_file = f"../models/PERL/results_{DataName}/{d}_{i}_convergence_rate.csv"

        if not (os.path.exists(data_driven_file) and os.path.exists(pinn_file) and os.path.exists(perl_file)):
            continue

        data_driven_data = np.loadtxt(data_driven_file, delimiter=",")
        pinn_data = np.loadtxt(pinn_file, delimiter=",")
        perl_data = np.loadtxt(perl_file, delimiter=",")

        data_driven_train_loss = data_driven_data[:, 0]
        pinn_train_loss = pinn_data[:, 0]
        perl_train_loss = perl_data[:, 0]

        data_driven_train_losses.append(data_driven_train_loss)
        pinn_train_losses.append(pinn_train_loss)
        perl_train_losses.append(perl_train_loss)

        # 记录最大训练损失长度
        max_train_length = max(max_train_length, len(data_driven_train_loss), len(pinn_train_loss),
                               len(perl_train_loss))

    if not data_driven_train_losses or not pinn_train_losses or not perl_train_losses:
        print("No data files found.")
        return

    # 填充或截断数据以确保所有模型的训练损失长度相同
    for train_losses in [data_driven_train_losses, pinn_train_losses, perl_train_losses]:
        for i in range(len(train_losses)):
            while len(train_losses[i]) < max_train_length:
                train_losses[i] = np.append(train_losses[i], train_losses[i][-1])  # 用最后一个值填充
            train_losses[i] = train_losses[i][:max_train_length]  # 截断至最大长度

    # 将列表转换为二维数组
    all_train_losses = np.array(data_driven_train_losses + pinn_train_losses + perl_train_losses)

    fig, ax = plt.subplots(figsize=(4, 3.2))

    # 1. 计算每个模型的损失的均值和标准差
    data_driven_mean = np.mean(data_driven_train_losses, axis=0)
    data_driven_std = np.std(data_driven_train_losses, axis=0)

    pinn_mean = np.mean(pinn_train_losses, axis=0)
    pinn_std = np.std(pinn_train_losses, axis=0)

    perl_mean = np.mean(perl_train_losses, axis=0)
    perl_std = np.std(perl_train_losses, axis=0)

    # 2. 绘制每个模型的透明阴影区域
    colors = ["#ff7700", "#7A00CC", "#0059b3"]
    plt.fill_between(range(max_train_length), data_driven_mean - data_driven_std,
                     data_driven_mean + data_driven_std, color=colors[0], alpha=0.2, label="Data Driven Shadow")
    plt.fill_between(range(max_train_length), pinn_mean - pinn_std,
                     pinn_mean + pinn_std, color=colors[1], alpha=0.2, label="PINN Shadow")
    plt.fill_between(range(max_train_length), perl_mean - perl_std,
                     perl_mean + perl_std, color=colors[2], alpha=0.2, label="PERL Shadow")

    # 3. 绘制每个模型的平均线
    plt.plot(data_driven_mean, label="NN", color=colors[0], linestyle='-.', linewidth=2)
    plt.plot(pinn_mean, label="PINN", color=colors[1], linestyle=':', linewidth=2)
    plt.plot(perl_mean, label="PERL", color=colors[2], linestyle='-', linewidth=2)

    # 其他设置
    plt.xlabel("Epoch")
    plt.xlim(0, max_train_length)

    ax.set_ylabel("MSE Loss $(m^2/s^4)$")
    ax.set_yscale("log")
    plt.ylim(0.0005, 10)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    plt.subplots_adjust(left=0.24, right=0.94, bottom=0.2, top=0.95)

    #plt.legend(loc='upper right', frameon=False, fontsize=12, ncol=2)

    plt.savefig(f'{d} Convergence Rate.png', dpi=350)
    #plt.show()



# Copy
for k in range(1,6):
    shutil.copy(f"../models/Data_driven/results_{DataName}/r0_500_{k}_convergence_rate.csv",
                f"../models/Data_driven/results_{DataName}/r2_500_{k}_convergence_rate.csv")
    shutil.copy(f"../models/Data_driven/results_{DataName}/r0_1000_{k}_convergence_rate.csv",
                f"../models/Data_driven/results_{DataName}/r2_1000_{k}_convergence_rate.csv")
    shutil.copy(f"../models/Data_driven/results_{DataName}/r0_5000_{k}_convergence_rate.csv",
                f"../models/Data_driven/results_{DataName}/r2_5000_{k}_convergence_rate.csv")
    shutil.copy(f"../models/Data_driven/results_{DataName}/r0_42000_{k}_convergence_rate.csv",
                f"../models/Data_driven/results_{DataName}/r2_42000_{k}_convergence_rate.csv")



# plot_data_single("r0_500",1,1,1) # Data-driven, PINN, PERL
# plot_data_single("r0_1000",1,1,1)
# plot_data_single("r0_5000",1,1,2)
# plot_data_single("r0_42000",1,1,1)

# plot_data_single("r1_500",4,1,1)
# plot_data_single("r1_1000",1,2,3)
# plot_data_single("r1_5000",2,1,1)
# plot_data_single("r1_42000",3,1,4)

# plot_data_single("r2_500",1,1,1)
# plot_data_single("r2_1000",1,1,1)
# plot_data_single("r2_5000",1,1,1)
# plot_data_single("r2_42000",1,1,1)


plot_data_single("r0_1000",2,3,2)
plot_data_single("r0_42000",4,3,2)
plot_data_single("r1_1000",1,4,1)
plot_data_single("r1_42000",4,1,1)