import sys
import os

oraingo_bidea = os.path.dirname(os.path.realpath(__file__))
bide_orokorra = os.path.dirname(oraingo_bidea)

from sklearn.svm import SVC
import pandas as pd
import numpy as np
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

# # Modeloa sortu:
modelo_gauss = SVC(C = 10, kernel = "rbf", gamma = 10**(-2))
t0 = time.time()
modelo_gauss.fit(X_entrenamendu, Y_entrenamendu)
t1 = time.time()
nota = modelo_gauss.score(X_test, Y_test)

# # Informazioa gorde:
modelo_gauss_info = []
modelo_gauss_info.append(t1-t0)
modelo_gauss_info.append(nota)

# # Informazioa inprimatu
print(f"Kernel gaussiarreko modeloaren nota: {nota}")
print(f"Kernel gaussiarreko modeloaren entrenamendu denbora: {modelo_gauss_info[0]}")

# # Modeloa gorde:
pickle.dump(modelo_gauss, open(os.path.join(bide_orokorra, "Entrenatutako_modeloak","Scikit_modelo_hoberena_gauss.pkl"), "wb"))
pickle.dump(modelo_gauss_info, open(os.path.join(bide_orokorra, "Entrenatutako_modeloak","Scikit_modelo_hoberena_gauss_info.pkl"), "wb"))






# # ----------------------------------------------------------------------------
# # ------------------MODELO HOBERENA: KERNEL POLINOMIALA-----------------------
# # ----------------------------------------------------------------------------

# # Modeloa sortu:
modelo_poly = SVC(C=10, kernel = "poly", degree = 6, coef0 =1)
t0 = time.time()
modelo_poly.fit(X_entrenamendu, Y_entrenamendu)
t1 = time.time()
nota = modelo_poly.score(X_test, Y_test)

# # Informazioa gorde:
modelo_poly_info = []
modelo_poly_info.append(t1-t0)
modelo_poly_info.append(nota)

# # Informazioa inprimatu
print(f"Kernel polinomialeko modeloaren nota: {nota}")
print(f"Kernel polinomialeko modeloaren entrenamendu denbora: {modelo_poly_info[0]}")

# # Modeloa gorde:
pickle.dump(modelo_poly, open(os.path.join(bide_orokorra, "Entrenatutako_modeloak","Scikit_modelo_hoberena_poly.pkl"), "wb"))
pickle.dump(modelo_poly_info, open(os.path.join(bide_orokorra, "Entrenatutako_modeloak","Scikit_modelo_hoberena_poly_info.pkl"), "wb"))

