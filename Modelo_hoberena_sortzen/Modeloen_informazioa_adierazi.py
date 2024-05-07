import sys
import os
import matplotlib.pyplot as plt

oraingo_bidea = os.path.dirname(os.path.realpath(__file__))
bide_orokorra = os.path.dirname(oraingo_bidea)
sys.path.insert(0,os.path.join(bide_orokorra, "Algoritmoak"))


import pandas as pd
import numpy as np
from SGD_soft_SVM_Kernels import Nire_SGD_kernelekin
import time
import pickle
import random

# # ----------------------------------------------------------------------------
# # -------------------------DATU BASEA INPORTATU-------------------------------
# # ----------------------------------------------------------------------------
mnist_test_bidea = os.path.join(bide_orokorra, "Datu_basea\mnist_test.csv")
mnist_train_bidea = os.path.join(bide_orokorra, "Datu_basea\mnist_train.csv")

entrenamendu_datuak = pd.read_csv(mnist_train_bidea)
testeatzeko_datuak = pd.read_csv(mnist_test_bidea)
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]
X_test = testeatzeko_datuak.iloc[:,1:]
Y_test = testeatzeko_datuak["label"]

# Transformazioa:
X_entrenamendu = X_entrenamendu / 255
X_test = X_test / 255




# # ----------------------------------------------------------------------------
# # --------------------NIRE MODELOA KERNEL GAUSSIARRA--------------------------
# # ----------------------------------------------------------------------------

# # Modeloa inportatu
Nire_gauss = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_gauss.pkl"),"rb"))
Nire_gauss_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_gauss_info.pkl"),"rb"))

# # Informazioa erakutsi:
print("Nire modeloa, kernel gausiarra:")
print(f"Behar izan duen denbora: {Nire_gauss_info[0]}")
print(f"Lortu duen asmatze-proportzioa: {Nire_gauss_info[1]}")
print()

# # Conffusion Matrix egin:

# Nire_gauss_test_predikzioak = Nire_gauss.predict_anitzkoitza(X_test.values())

# # Gaizki egindako predikzioak bisualizatu:
digitu_kopurua = 10
fig, axs = plt.subplots(2,5, figsize = (12,6))
k = 0
for i in range(2):
    for j in range(5):
        while Nire_gauss.predict(X_test.iloc[k,:]) != Y_test.iloc[i]:
            k += 1
        adib = np.array(X_test.iloc[k, :])
        adib = np.reshape(adib, (28,28))
        axs[i,j].imshow(adib, cmap = "gray_r")
        axs[i,j].axis("off")
        k += 1

plt.tight_layout(pad = 0.2)
plt.show()




# # ----------------------------------------------------------------------------
# # --------------------NIRE MODELOA KERNEL POLINOMIALA-------------------------
# # ----------------------------------------------------------------------------

# # Modeloa inportatu
Nire_poly = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_poly.pkl"),"rb"))
Nire_poly_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_poly_info.pkl"),"rb"))

# # Informazioa erakutsi:

print("Nire modeloa, kernel polinomiala:")
print(f"Behar izan duen denbora: {Nire_poly_info[0]}")
print(f"Lortu duen asmatze-proportzioa: {Nire_poly_info[1]}")
print()






# # # ----------------------------------------------------------------------------
# # # --------------------SCIKIT MODELOA KERNEL GAUSSIARRA--------------------------
# # # ----------------------------------------------------------------------------

# # # Modeloa inportatu
# Scikit_gauss = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_gauss.pkl"),"rb"))
# Scikit_gauss_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_gauss_info.pkl"),"rb"))

# # # Informazioa erakutsi:
# print("Nire modeloa, kernel gausiarra:")
# print(f"Behar izan duen denbora: {Scikit_gauss_info[0]}")
# print(f"Lortu duen asmatze-proportzioa: {Scikit_gauss_info[1]}")
# print()

# # Conffusion Matrix egin:






# # # ----------------------------------------------------------------------------
# # # --------------------SCIKIT MODELOA KERNEL POLINOMIALA-------------------------
# # # ----------------------------------------------------------------------------

# # # Modeloa inportatu
# Scikit_poly = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_poly.pkl"),"rb"))
# Scikit_poly_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_poly_info.pkl"),"rb"))

# # # Informazioa erakutsi:

# print("Nire modeloa, kernel polinomiala:")
# print(f"Behar izan duen denbora: {Scikit_poly_info[0]}")
# print(f"Lortu duen asmatze-proportzioa: {Scikit_poly_info[1]}")
# print()

