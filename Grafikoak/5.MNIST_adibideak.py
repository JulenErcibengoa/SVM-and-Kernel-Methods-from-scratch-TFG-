import sys
import os

oraingo_bidea = os.path.dirname(os.path.realpath(__file__))
bide_orokorra = os.path.dirname(oraingo_bidea)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mnist_train_bidea = os.path.join(bide_orokorra, "Datu_basea\mnist_train.csv")
entrenamendu_datuak = pd.read_csv(mnist_train_bidea)
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]

fig, axs = plt.subplots(2, 5, figsize=(12, 6))
k = 0
digitua = 0
for i in range(2):
    for j in range(5):
        while Y_entrenamendu[k] != digitua:
            k+=1
        adib = np.array(X_entrenamendu.iloc[k, :])
        adib = np.reshape(adib, (28, 28))
        axs[i, j].imshow(adib, cmap="gray_r")
        axs[i, j].axis('off') 
        digitua += 1

plt.tight_layout(pad = 0.2)
plt.show()

