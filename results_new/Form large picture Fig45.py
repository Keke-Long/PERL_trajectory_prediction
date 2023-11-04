from PIL import Image, ImageDraw, ImageFont

r_num = 0 #0用于Fig4, 1用于Fig5

# 创建一个空白的图像，用于拼接
img_width = 1400
img_height = 1400
img = Image.new('RGB', (img_width*4+80, img_height+220), (255, 255, 255))

# 贴图
h = 185
img.paste(Image.open(f"../results/r{r_num}_one-step a.png"), (20, h))
img.paste(Image.open(f"../results/r{r_num}_one-step v.png"), (20+img_width, h))
h = 350
img.paste(Image.open(f"../results/r{r_num}_1000 Convergence Rate new.png"), (20+img_width*2+40, h))
img.paste(Image.open(f"../results/r{r_num}_42000 Convergence Rate new.png"),(20+img_width*3+60, h))


#加legend图
legend = Image.open('../results/legend MSE.png')
legend = legend.resize((int(legend.width * 0.9), int(legend.height * 0.9)))
img.paste(legend, (250, 10))

legend = Image.open('../results/legend convergence1.png')
legend = legend.resize((int(legend.width*1.15), int(legend.height*1.15)))
img.paste(legend, (3060, 70))

legend = Image.open('../results/legend convergence2.png')
legend = legend.resize((int(legend.width*1.15), int(legend.height*1.15)))
img.paste(legend, (3500, 70))


# 添加序号
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 130)  # 可以更改字体和大小
draw.text((30, 15), "A", font=font, fill='black')
draw.text((30+img_width, 15), "B", font=font, fill='black')
draw.text((20+img_width*2+60, 15), "C", font=font, fill='black')
draw.text((20+img_width*3+65, 15), "D", font=font, fill='black')

h = 1480
font = ImageFont.truetype("arial.ttf", 100)  # 可以更改字体和大小
draw.text((3200, h), "Train data size = 500", font=font, fill='black')
draw.text((4600, h), "Train data size = 20000", font=font, fill='black')

# 画框
h = img_height+210
draw.rectangle([10, 10,             15+2*img_width, h], outline='silver', width=5)
draw.rectangle([45+2*img_width, 10, 70+4*img_width, h], outline='silver', width=5)

# 保存合并后的图像
img.save(f"Fig {r_num+4}.png", dpi=(350, 350))