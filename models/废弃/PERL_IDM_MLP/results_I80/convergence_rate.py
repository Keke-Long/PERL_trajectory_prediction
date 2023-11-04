import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("convergence_rate.csv")

train_loss = data['train_loss']
val_loss = data['val_loss']

plt.figure(figsize=(7, 4))
plt.plot(train_loss, label="Train Loss", color='blue')
plt.plot(val_loss, label="Validation Loss", color='red')

#plt.title("Training and Validation Loss Convergence")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()