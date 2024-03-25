import pandas as pd
import numpy as np
from SGD_soft_SVM_Kernels import Nire_SGD_kernelekin
import time
import pickle



# -------------------------------------------------------------------------------
# -----------------------------DATU BASEA INPORTATU------------------------------
# -------------------------------------------------------------------------------
entrenamendu_datuak = pd.read_csv("mnist_train.csv")
testeatzeko_datuak = pd.read_csv("mnist_test.csv")
# print(entrenamendu_datuak.head())
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]

X_test = testeatzeko_datuak.iloc[:,1:]
Y_test = testeatzeko_datuak["label"]



# -------------------------------------------------------------------------------
# -------------------------DATU BASEA 0-1 tartean jarri--------------------------
# -------------------------------------------------------------------------------
X_entrenamendu = X_entrenamendu / 255
X_test = X_test / 255



# -------------------------------------------------------------------------------
# ------------------------MODELOA ENTRENATU ETA GORDE----------------------------
# -------------------------------------------------------------------------------

iterazio_kopurua = 10000
modeloa = Nire_SGD_kernelekin(4,"kernel gaussiarra")
h_time = time.time()
modeloa.fit(X_entrenamendu.values,Y_entrenamendu.values,iterazio_kopurua)
end_time = time.time()
print(f"Behar izan duen denbora {iterazio_kopurua} iteraziorekin {end_time-h_time}s da")
pickle.dump(modeloa,open("Nire_SVC_modeloa_MNIST_iter10000.pkl","wb"))





# Behar izan duen denbora = 3760.797991514206 segundu

# -------------------------------------------------------------------------------
# -----------------------------MODELOA INPORTATU---------------------------------
# -------------------------------------------------------------------------------

modeloa = pickle.load(open("Nire_SVC_modeloa_MNIST_iter10000.pkl","rb"))
nota = modeloa.score(X_test.values,Y_test.values)

informazioa = [f"Behar izan duen denbora = {end_time-h_time}",f"Modeloaren nota = {nota}"]
pickle.dump(informazioa,open("Informazioa_10000.pkl","wb"))