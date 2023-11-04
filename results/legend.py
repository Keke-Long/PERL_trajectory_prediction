import matplotlib.pyplot as plt
from PIL import Image

# # 创建一个图形
# fig, ax = plt.subplots(figsize=(8, 4))
#
# # 假设这是您的图形绘制函数，包括四条线条
# # 这里只是示例，您需要将您的绘图函数替换为适当的代码
# line1, = ax.plot([1, 2, 13], [2, 0, 0], label='Line 1', linestyle=':', color='#505050', marker='d', markersize=5, markerfacecolor='none', linewidth=2)
# line2, = ax.plot([1, 2, 13], [3, 0, 0], label='Line 2', linestyle=':', color='#ff7700', marker='s', markersize=5, markerfacecolor='none', linewidth=2)
# line3, = ax.plot([1, 2, 3], [1, 0, 0], label='Line 3', linestyle='--', color='#7A00CC', marker='^', markersize=5, linewidth=2)
# line4, = ax.plot([1, 2, 3], [30, 0, 0], label='Line 4', linestyle='-', color='#0059b3', marker='o', markersize=4, linewidth=2)
#
# # 创建自定义图例项
# legend_labels = ['Physics', 'NN', 'PINN', 'PERL']
# legend_handles = [line1, line2, line3, line4]
#
# # 添加数据标签（图例）
# legend = ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize=18, frameon=False, ncol=2)
#
# # 保存整个图形
# plt.savefig("plot_with_custom_legend.png", dpi=350)



# # 打开图像文件
# image = Image.open("plot_with_custom_legend.png")
#
# # 指定要截取的区域的坐标（左上角和右下角的坐标）
# left = 1220  # 左上角的横坐标
# top = 210   # 左上角的纵坐标
# right = left+1230  # 右下角的横坐标
# bottom = top+240  # 右下角的纵坐标
#
# # 使用crop方法截取指定区域
# cropped_image = image.crop((left, top, right, bottom))
#
# # 保存截取的图像为新文件
# cropped_image.save("legend MSE.png")
#
# # 关闭图像对象
# image.close()



# 打开图像文件
image = Image.open("r0_500 Convergence Rate1.png")

# # 指定要截取的区域的坐标（左上角和右下角的坐标）
# left = 3310  # 左上角的横坐标
# top = 280   # 左上角的纵坐标
# right = left+320  # 右下角的横坐标
# bottom = top+225  # 右下角的纵坐标
# cropped_image = image.crop((left, top, right, bottom))
# #cropped_image.show()
# cropped_image.save("legend convergence1.png")
# image.close()

# 指定要截取的区域的坐标（左上角和右下角的坐标）
left = 3775  # 左上角的横坐标
top = 280   # 左上角的纵坐标
right = left+680  # 右下角的横坐标
bottom = top+235  # 右下角的纵坐标
cropped_image = image.crop((left, top, right, bottom))
cropped_image.save("legend convergence2.png")
image.close()