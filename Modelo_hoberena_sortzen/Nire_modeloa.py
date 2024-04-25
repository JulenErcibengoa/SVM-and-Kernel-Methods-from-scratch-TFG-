import sys
import os

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
# # -------------------MODELO HOBERENA: KERNEL GAUSSIARRA-----------------------
# # ----------------------------------------------------------------------------

# # # Modeloa sortu:
# modelo_gauss = Nire_SGD_kernelekin(koeficient=10**(-3), kernel = "kernel gaussiarra", sigma = 10)
# t0 = time.time()
# random.seed(123)
# modelo_gauss.fit(X_entrenamendu.values, Y_entrenamendu.values, iter = 10000)
# t1 = time.time()
# nota = modelo_gauss.score(X_test.values, Y_test.values)

# # Informazioa gorde:
# modelo_gauss_info = []
# modelo_gauss_info.append(t1-t0)
# modelo_gauss_info.append(nota)

# # Informazioa inprimatu
# print(f"Kernel gaussiarreko modeloaren nota: {nota}")
# print(f"Kernel gaussiarreko modeloaren entrenamendu denbora: {modelo_gauss_info[0]}")

# # # Modeloa gorde:
# pickle.dump(modelo_gauss, open(os.path.join(bide_orokorra, "Entrenatutako_modeloak","Nire_modelo_hoberena_gauss.pkl"), "wb"))
# pickle.dump(modelo_gauss_info, open(os.path.join(bide_orokorra, "Entrenatutako_modeloak","Nire_modelo_hoberena_gauss_info.pkl"), "wb"))






# # ----------------------------------------------------------------------------
# # ------------------MODELO HOBERENA: KERNEL POLINOMIALA-----------------------
# # ----------------------------------------------------------------------------

# # Modeloa sortu:
modelo_poly = Nire_SGD_kernelekin(koeficient=10**(3), kernel = "kernel polinomiala", deg = 3)
t0 = time.time()
random.seed(123)
modelo_poly.fit(X_entrenamendu.values, Y_entrenamendu.values, iter = 10000)
t1 = time.time()
nota = modelo_poly.score(X_test.values, Y_test.values)

# Informazioa gorde:
modelo_poly_info = []
modelo_poly_info.append(t1-t0)
modelo_poly_info.append(nota)

# Informazioa inprimatu
print(f"Kernel polinomialeko modeloaren nota: {nota}")
print(f"Kernel polinomialeko modeloaren entrenamendu denbora: {modelo_poly_info[0]}")

# # Modeloa gorde:
pickle.dump(modelo_poly, open(os.path.join(bide_orokorra, "Entrenatutako_modeloak","Nire_modelo_hoberena_poly.pkl"), "wb"))
pickle.dump(modelo_poly_info, open(os.path.join(bide_orokorra, "Entrenatutako_modeloak","Nire_modelo_hoberena_poly_info.pkl"), "wb"))



