from PIL import Image, ImageDraw, ImageFont

r_num = 1

# 创建一个空白的图像，用于拼接
img_width = 1400
img_height = 1400
img = Image.new('RGB', (img_width*4+80, img_height*2+360), (255, 255, 255))

# 贴图
h = 185
img.paste(Image.open(f"r{r_num}_one-step a.png"), (20, h))
img.paste(Image.open(f"r{r_num}_one-step v.png"), (20+img_width, h))
img.paste(Image.open(f"r{r_num}_multi-step a.png"), (20+img_width*2+60, h))
img.paste(Image.open(f"r{r_num}_multi-step v.png"), (20+img_width*3+60, h))

h = 520 + img_height
img.paste(Image.open(f"r{r_num}_500 Convergence Rate.png"), (20, h))
img.paste(Image.open(f"r{r_num}_1000 Convergence Rate.png"), (20+img_width+20, h))
img.paste(Image.open(f"r{r_num}_5000 Convergence Rate.png"), (20+img_width*2+40, h))
img.paste(Image.open(f"r{r_num}_42000 Convergence Rate.png"),(20+img_width*3+60, h))


#加legend图
legend = Image.open('legend MSE.png')
legend = legend.resize((int(legend.width * 0.9), int(legend.height * 0.9)))
img.paste(legend, (250, 10))

legend = Image.open('legend convergence1.png')
legend = legend.resize((int(legend.width*1.1), int(legend.height*1.1)))
img.paste(legend, (125, 300 + img_height))

legend = Image.open('legend convergence2.png')
legend = legend.resize((int(legend.width*1.1), int(legend.height*1.1)))
img.paste(legend, (800, 300 + img_height))


# 添加序号
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 130)  # 可以更改字体和大小
draw.text((30, 15), "A", font=font, fill='black')
draw.text((30+img_width, 15), "B", font=font, fill='black')
draw.text((20+img_width*2+60, 15), "C", font=font, fill='black')
draw.text((20+img_width*3+60, 15), "D", font=font, fill='black')

h = 250+img_height
draw.text((30, h), "E", font=font, fill='black')
draw.text((60+img_width, h), "F", font=font, fill='black')
draw.text((20+img_width*2+60, h), "G", font=font, fill='black')
draw.text((50+img_width*3+60, h), "H", font=font, fill='black')

font = ImageFont.truetype("arial.ttf", 100)  # 可以更改字体和大小
h = 1610+img_height
draw.text((380, h), "Train data size = 300", font=font, fill='black')
draw.text((380+img_width, h), "Train data size = 600", font=font, fill='black')
draw.text((380+img_width*2+30, h), "Train data size = 3000", font=font, fill='black')
draw.text((380+img_width*3+20, h), "Train data size = 25200", font=font, fill='black')

# 画框
h = img_height+210
draw.rectangle([10, 10,             15+2*img_width, h], outline='silver', width=5)
draw.rectangle([45+2*img_width, 10, 70+4*img_width, h], outline='silver', width=5)
draw.rectangle([10, h+30, 70+4*img_width, img_height*2+350], outline='silver', width=5)

# 保存合并后的图像
img.save(f"r{r_num} merged.png", dpi=(350, 350))