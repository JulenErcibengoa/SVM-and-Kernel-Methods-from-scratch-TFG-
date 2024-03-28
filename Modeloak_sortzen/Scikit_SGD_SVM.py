from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import time
from sklearn.kernel_approximation import Nystroem


# Datu basea:
entrenamendu_datuak = pd.read_csv("mnist_train.csv")
testeatzeko_datuak = pd.read_csv("mnist_test.csv")
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]
X_test = testeatzeko_datuak.iloc[:,1:]
Y_test = testeatzeko_datuak["label"]

# SGD_SVM

X_entrenamendu = X_entrenamendu 
X_test = X_test 


Y_zero_train = [1 if elementua == 0 else -1 for elementua in Y_entrenamendu.values]
Y_zero_test = [1 if elementua == 0 else -1 for elementua in Y_test.values]


feature_map_nystroem = Nystroem(gamma = 0.01)
X_entrenamendu = feature_map_nystroem.fit_transform(X_entrenamendu)
X_test = feature_map_nystroem.fit_transform(X_test)

max_iter = 10000
modeloa = SGDOneClassSVM(max_iter=max_iter,nu = 1,tol = 10**(-6))

hasiera = time.time()
modeloa.fit(X_entrenamendu,Y_zero_train)
bukaera = time.time()

balioak = modeloa.predict(X_test)
scores = modeloa.score_samples(X_test)



nota1 = 0
nota2 = 0
for i,balio in enumerate(balioak):
    if balio == Y_zero_test[i]: 
        nota1 += 1

    if scores[i] > 0 and Y_zero_test[i] == 1:
        nota2 += 1

print(nota1)
print(nota2)
print(scores)
print(len(Y_zero_test)  )
nota1 = nota1 / len(Y_zero_test)  

print(f"Nota = {nota1}\nEntrenamendu denbora = {bukaera-hasiera}")

print(modeloa.get_params())