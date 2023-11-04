import subprocess
from datetime import datetime

with open(f"./results_NGSIM_US101/predict_MSE_results.txt", 'a') as f:
    f.write('\n')
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    f.write(f'{current_time}\n')
    f.write(f'num_samples = 500 lstm+Newell\n')

# 循环运行
for _ in range(5):
    subprocess.run(["python", "train.py"])
