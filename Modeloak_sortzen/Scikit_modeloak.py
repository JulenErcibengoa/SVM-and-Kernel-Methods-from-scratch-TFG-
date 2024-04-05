from sklearn.svm import SVC
import numpy as np
import pandas as pd
import time
import pickle
from joblib import Parallel,delayed
import matplotlib.pyplot as plt





# ----------------------------------------------------------------------------
# -------------------------DATU BASEA INPORTATU-------------------------------
# ----------------------------------------------------------------------------
entrenamendu_datuak = pd.read_csv("mnist_train.csv")
testeatzeko_datuak = pd.read_csv("mnist_test.csv")
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]
X_test = testeatzeko_datuak.iloc[:,1:]
Y_test = testeatzeko_datuak["label"]
X_entrenamendu = X_entrenamendu / 255
X_test = X_test / 255
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# -----INFORMAZIOA GORDETZEKO LEKUAK SORTU (LEHENENGO ALDIZ EJEKUTATZEAN)-----
# ----------------------------------------------------------------------------
# Hau lehenengo aldiz exekutatzean bakarrik exekutatu, bestela informazioa galdu egingo da

# Notak_matrizea_rbf = np.zeros((7,7))
# Notak_matrizea_poly = np.zeros((7,7))

# pickle.dump(Notak_matrizea_rbf,open("Notak_matrizea_rbf.pkl","wb"))
# pickle.dump(Notak_matrizea_poly,open("Notak_matrizea_poly.pkl","wb"))

# with open('Modeloen_notak.txt', 'w') as informazioa:
#     informazioa.write("Scikit-Learn modelo desberdinak, kernel gaussiarra\n\n")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# -------INFORMAZIOA GORDETZEKO LEKUAK IREKI (LEHENENGO ITERAZIOAN EZ)--------
# ----------------------------------------------------------------------------

Notak_matrizea_rbf = pickle.load(open("Notak_matrizea_rbf.pkl","rb"))
Notak_matrizea_poly = pickle.load(open("Notak_matrizea_poly.pkl","rb"))
print(f"INFORMAZIOA KARGATUTA: \n Noten matrizea kernel gaussiarra = \n{Notak_matrizea_rbf}\n Noten matrizea kernel polinomiala = \n{Notak_matrizea_poly}")

plt.imshow(Notak_matrizea_rbf)
plt.xticks(np.arange(0, 7, 1), [r'$10^{{{}}}$'.format(j) for j in [-3,-2,-1,0,1,2,3]])
plt.xlabel("gamma")
plt.yticks(np.arange(0, 7, 1), [r'$10^{{{}}}$'.format(j) for j in [-3,-2,-1,0,1,2,3]])
plt.ylabel("C", rotation = 0)
plt.title("Modelo desberdinen asmatze proportzioa baliozta-multzoan:\nKernel gaussiarra")
plt.colorbar(label='Asmatutako proportzioa')
plt.tight_layout(pad = 0.2)
for i in range(Notak_matrizea_rbf.shape[0]):
    for j in range(Notak_matrizea_rbf.shape[1]):
        plt.text(j, i, '{:.3f}'.format(Notak_matrizea_rbf[i, j]), ha='center', va='center', color='white' if Notak_matrizea_rbf[i, j] < 0.5 else "black")
plt.show()



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------



# Entrenamendu asko egiteko aldi berean (joblib): 
def entrenatu (C,param,kernela,i,j,Nota_matrizea):
    if kernela == "rbf":
        if Nota_matrizea[i,j] == 0:
            print(f"Hasi da ({i},{j}) posizioa, C = {C}, gamma = {param}")

            modeloa = SVC(C=C,kernel="rbf",gamma=param)
            modeloa.fit(X_entrenamendu,Y_entrenamendu)
            nota = modeloa.score(X_test,Y_test)
            Nota_matrizea[i][j] = nota
            with open('Modeloen_notak.txt', 'a') as informazioa:
                informazioa.write(f"(i,j) = ({i},{j}), C = {C}, gamma = {param} --> Nota = {Nota_matrizea[i,j]}\n")
            
            print(f"Eginda ({i},{j}) posizioaren entrenamendua, horrela gelditu da noten matrizea:")
            print(Nota_matrizea)
        else:
            print(f"Modeloa ({i},{j}) posizioan jada entrenatua izan da:\nC = {C}, gamma = {param}, nota = {Nota_matrizea[i,j]}")
    
    elif kernela == "poly":
        if Nota_matrizea[i,j] == 0:
            print(f"Hasi da ({i},{j}) posizioa, C = {C}, maila = {param}")

            modeloa = SVC(C=C,kernel="poly",degree=param)
            modeloa.fit(X_entrenamendu,Y_entrenamendu)
            nota = modeloa.score(X_test,Y_test)
            Nota_matrizea[i][j] = nota

            with open('Modeloen_notak.txt', 'a') as informazioa:
                informazioa.write(f"(i,j) = ({i},{j}), C = {C}, maila = {param} --> Nota = {Nota_matrizea[i,j]}\n")
            print(f"Eginda ({i},{j}) posizioaren entrenamendua, horrela gelditu da noten matrizea:")
            print(Nota_matrizea)
        else:
            print(f"Modeloa ({i},{j}) posizioan jada entrenatua izan da:\nC = {C}, maila = {param}, nota = {Nota_matrizea[i,j]}")



# ----------------------------------------------------------------------------
# ----------------------------KERNEL GAUSSIARRA-------------------------------
# ----------------------------------------------------------------------------

# Parametroak:
C_parametroak = np.logspace(-3, 3, 7) 
gamma_parametroak = np.logspace(-3, 3, 7)

# Entrenatu (entrenamendua geldi daiteke, baina goiko kodea komentatu egin behar da berriro hasterakoan entrenatzen,
# bestela informazioa galdu egingo da)
for i,C in enumerate(C_parametroak):
    for j,gamma in enumerate(gamma_parametroak):
        Notak_matrizea_rbf = pickle.load(open("Notak_matrizea_rbf.pkl","rb"))
        entrenatu(C,gamma,"rbf",i,j,Notak_matrizea_rbf)
        pickle.dump(Notak_matrizea_rbf,open("Notak_matrizea_rbf.pkl","wb"))


# ------------------------------GRAFIKOA EGIN---------------------------------
# Notak_matrizea_rbf = pickle.load(open("Notak_matrizea_rbf.pkl","rb"))
# plt.imshow(Notak_matrizea_rbf)
# plt.xticks(np.arange(0, 7, 1), [r'$10^{{{}}}$'.format(j) for j in [-3,-2,-1,0,1,2,3]])
# plt.xlabel("gamma")
# plt.yticks(np.arange(0, 7, 1), [r'$10^{{{}}}$'.format(j) for j in [-3,-2,-1,0,1,2,3]])
# plt.ylabel("C", rotation = 0)
# plt.title("Modelo desberdinen asmatze proportzioa baliozta-multzoan:\nKernel gaussiarra")
# plt.colorbar(label='Asmatutako proportzioa')
# plt.tight_layout(pad = 0.2)
# for i in range(Notak_matrizea_rbf.shape[0]):
#     for j in range(Notak_matrizea_rbf.shape[1]):
#         plt.text(j, i, '{:.3f}'.format(Notak_matrizea_rbf[i, j]), ha='center', va='center', color='white' if Notak_matrizea_rbf[i, j] < 0.5 else "black")
# plt.show()














# ----------------------------------------------------------------------------
# ----------------------------KERNEL POLINOMIALA------------------------------
# ----------------------------------------------------------------------------
        
# with open('Modeloen_notak.txt', 'a') as informazioa:
#     informazioa.write("\n\nScikit-Learn modelo desberdinak, kernel polinomiala\n\n")

# # Parametroak:
# C_parametroak = np.logspace(-3, 3, 7)  # Desde 0.01 hasta 1000, con 7 valores en escala logarítmica
# maila_desberdinak = [1,2,3,5,7,9,11]

# # Entrenatu (entrenamendua geldi daiteke, baina goiko kodea komentatu egin behar da berriro hasterakoan entrenatzen,
# # bestela informazioa galdu egingo da)
# for i,C in enumerate(C_parametroak):
#     for j,d in enumerate(maila_desberdinak):
#         Notak_matrizea_poly = pickle.load(open("Notak_matrizea_rbf.pkl","rb"))
#         entrenatu(C,d,"poly",i,j,Notak_matrizea_poly)
#         pickle.dump(Notak_matrizea_poly,open("Notak_matrizea_rbf.pkl","wb"))


















with open('Modeloen_notak.txt', 'r') as informazioa:
    info = informazioa.read()
    print(info)