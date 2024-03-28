from sklearn.svm import SVC
import numpy as np
import pandas as pd
import time
import pickle

# Datu basea:
entrenamendu_datuak = pd.read_csv("mnist_train.csv")
testeatzeko_datuak = pd.read_csv("mnist_test.csv")
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]
X_test = testeatzeko_datuak.iloc[:,1:]
Y_test = testeatzeko_datuak["label"]

X_entrenamendu = X_entrenamendu / 255
X_test = X_test / 255


# Kernel gaussiarra

parametros_C = np.logspace(-3, 3, 7)  # Desde 0.01 hasta 1000, con 7 valores en escala logarítmica
parametros_gamma = np.logspace(-3, 3, 7)

Nota_matrizea = np.zeros((7,7))

print("Hasi da")
for i,C in enumerate(parametros_C):
    for j,gamma in enumerate(parametros_gamma):
        model = SVC(C = C, gamma= gamma, kernel="rbf")
        hasiera = time.time()
        model.fit(X_entrenamendu,Y_entrenamendu)
        bukaera = time.time()
        Nota_matrizea[i,j] = model.score(X_test,Y_test)
        print(f"i = {i},j = {j}, entrenamendu denbora = {bukaera-hasiera}, Nota = \n{Nota_matrizea}")

pickle.dump(Nota_matrizea,open("Nota_matrizea_rbf.pkl","wb"))

