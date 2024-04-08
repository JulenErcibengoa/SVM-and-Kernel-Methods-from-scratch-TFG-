from sklearn.svm import SVC
import numpy as np
import pandas as pd
import time
import pickle
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
print("Entrenamenduko klaseen agertze proportzioa:")
print(Y_entrenamendu.value_counts()/len(Y_entrenamendu))
print()
print("Testeko klaseen agertze proportzioa:")
print(Y_test.value_counts()/len(Y_test))
print()
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# -----INFORMAZIOA GORDETZEKO LEKUAK SORTU (LEHENENGO ALDIZ EJEKUTATZEAN)-----
# ----------------------------------------------------------------------------
# Hau lehenengo aldiz exekutatzean bakarrik exekutatu, bestela informazioa galdu egingo da

# Notak_matrizea_rbf = np.zeros((7,7))
# Notak_matrizea_poly = np.zeros((7,7))
# Notak_matrizea_poly_handia = np.zeros((21,19))

# pickle.dump(Notak_matrizea_rbf,open("Notak_matrizea_rbf.pkl","wb"))
# pickle.dump(Notak_matrizea_poly,open("Notak_matrizea_poly.pkl","wb"))
# pickle.dump(Notak_matrizea_poly_handia,open("Notak_matrizea_poly_handia.pkl","wb"))

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
Notak_matrizea_poly_handia = pickle.load(open("Notak_matrizea_poly_handia.pkl","rb"))
# Notak_matrizea_poly_handia[7:14,0:7] = Notak_matrizea_poly
# pickle.dump(Notak_matrizea_poly_handia,open("Notak_matrizea_poly_handia.pkl","wb"))
print(f"INFORMAZIOA KARGATUTA: \n\n Noten matrizea kernel gaussiarra = \n{Notak_matrizea_rbf}\n\n Noten matrizea kernel polinomiala = \n{Notak_matrizea_poly}\n\nNoten matrizea kernel polinomiala handia = \n{Notak_matrizea_poly_handia}\n\n")

# Notak_matrizea_poly_handia[7:14,0:7] = Notak_matrizea_poly
# pickle.dump(Notak_matrizea_poly_handia,open("Notak_matrizea_poly_handia.pkl","wb"))

plt.imshow(Notak_matrizea_poly_handia)
plt.xticks(np.arange(0, 19, 1), [i for i in range(2,21)])
plt.xlabel("polinomioaren maila")
plt.yticks(np.arange(0, 21, 1), [r'$10^{{{}}}$'.format(j) for j in range(-10,11)])
plt.ylabel("C", rotation = 0)
plt.title("Modelo desberdinen asmatze proportzioa baliozta-multzoan:\nKernel polinomiala: bertsio handia")
plt.colorbar(label='Asmatutako proportzioa')
plt.clim(0,1)
plt.tight_layout(pad = 0.2)
# for i in range(Notak_matrizea_poly_handia.shape[0]):
#     for j in range(Notak_matrizea_poly_handia.shape[1]):
#         plt.text(j, i, '{:.1f}'.format(Notak_matrizea_poly_handia[i, j]), ha='center', va='center', color='white' if Notak_matrizea_poly_handia[i, j] < 0.5 else "black")
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

            modeloa = SVC(C=C,kernel="poly",degree=param,coef0=1)
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
Notak_matrizea_rbf = pickle.load(open("Notak_matrizea_rbf.pkl","rb"))
plt.imshow(Notak_matrizea_rbf)
plt.xticks(np.arange(0, 7, 1), [r'$10^{{{}}}$'.format(j) for j in [-3,-2,-1,0,1,2,3]])
plt.xlabel("gamma")
plt.yticks(np.arange(0, 7, 1), [r'$10^{{{}}}$'.format(j) for j in [-3,-2,-1,0,1,2,3]])
plt.ylabel("C", rotation = 0)
plt.title("Modelo desberdinen asmatze proportzioa baliozta-multzoan:\nKernel gaussiarra")
plt.colorbar(label='Asmatutako proportzioa')
plt.clim(0,1)
plt.tight_layout(pad = 0.2)
for i in range(Notak_matrizea_rbf.shape[0]):
    for j in range(Notak_matrizea_rbf.shape[1]):
        plt.text(j, i, '{:.3f}'.format(Notak_matrizea_rbf[i, j]), ha='center', va='center', color='white' if Notak_matrizea_rbf[i, j] < 0.5 else "black")
plt.show()














# ----------------------------------------------------------------------------
# ----------------------------KERNEL POLINOMIALA------------------------------
# ----------------------------------------------------------------------------
        
# with open('Modeloen_notak.txt', 'a') as informazioa:
#     informazioa.write("\n\nScikit-Learn modelo desberdinak, kernel polinomiala\n\n")

# Parametroak:
C_parametroak = np.logspace(-3, 3, 7) 
maila_desberdinak = [2,3,4,5,6,7,8]

# Entrenatu (entrenamendua geldi daiteke, baina goiko kodea komentatu egin behar da berriro hasterakoan entrenatzen,
# bestela informazioa galdu egingo da)
for i,C in enumerate(C_parametroak):
    for j,d in enumerate(maila_desberdinak):
        Notak_matrizea_poly = pickle.load(open("Notak_matrizea_poly.pkl","rb"))
        entrenatu(C,d,"poly",i,j,Notak_matrizea_poly)
        pickle.dump(Notak_matrizea_poly,open("Notak_matrizea_poly.pkl","wb"))


# ------------------------------GRAFIKOA EGIN---------------------------------
Notak_matrizea_poly = pickle.load(open("Notak_matrizea_poly.pkl","rb"))
plt.imshow(Notak_matrizea_poly)
plt.xticks(np.arange(0, 7, 1), [2,3,4,5,6,7,8])
plt.xlabel("polinomioaren maila")
plt.yticks(np.arange(0, 7, 1), [r'$10^{{{}}}$'.format(j) for j in [-3,-2,-1,0,1,2,3]])
plt.ylabel("C", rotation = 0)
plt.title("Modelo desberdinen asmatze proportzioa baliozta-multzoan:\nKernel polinomiala")
plt.colorbar(label='Asmatutako proportzioa')
plt.clim(0,1)
plt.tight_layout(pad = 0.2)
for i in range(Notak_matrizea_poly.shape[0]):
    for j in range(Notak_matrizea_poly.shape[1]):
        plt.text(j, i, '{:.3f}'.format(Notak_matrizea_poly[i, j]), ha='center', va='center', color='white' if Notak_matrizea_poly[i, j] < 0.5 else "black")
plt.show()








# ----------------------------------------------------------------------------
# -------------------------KERNEL POLINOMIAL HANDIA---------------------------
# ----------------------------------------------------------------------------
        
# with open('Modeloen_notak.txt', 'a') as informazioa:
#     informazioa.write("\n\nScikit-Learn modelo desberdinak, kernel polinomiala: bertsio handitua\n\n")

# Parametroak:
C_parametroak = np.logspace(-10, 10, 21) 
maila_desberdinak = [i for i in range(2,21)]

# Entrenatu (entrenamendua geldi daiteke, baina goiko kodea komentatu egin behar da berriro hasterakoan entrenatzen,
# bestela informazioa galdu egingo da)
for i,C in enumerate(C_parametroak):
    for j,d in enumerate(maila_desberdinak):
        Notak_matrizea_poly_handia = pickle.load(open("Notak_matrizea_poly_handia.pkl","rb"))
        entrenatu(C,d,"poly",i,j,Notak_matrizea_poly_handia)
        pickle.dump(Notak_matrizea_poly_handia,open("Notak_matrizea_poly_handia.pkl","wb"))


# # ------------------------------GRAFIKOA EGIN---------------------------------
Notak_matrizea_poly_handia = pickle.load(open("Notak_matrizea_poly_handia.pkl","rb"))
plt.imshow(Notak_matrizea_poly_handia)
plt.xticks(np.arange(0, 19, 1), [i for i in range(2,21)])
plt.xlabel("polinomioaren maila")
plt.yticks(np.arange(0, 21, 1), [r'$10^{{{}}}$'.format(j) for j in range(-10,11)])
plt.ylabel("C", rotation = 0)
plt.title("Modelo desberdinen asmatze proportzioa baliozta-multzoan:\nKernel polinomiala: bertsio handia")
plt.colorbar(label='Asmatutako proportzioa')
plt.clim(0,1)
plt.tight_layout(pad = 0.2)
# for i in range(Notak_matrizea_poly_handia.shape[0]):
#     for j in range(Notak_matrizea_poly_handia.shape[1]):
#         plt.text(j, i, '{:.1f}'.format(Notak_matrizea_poly_handia[i, j]), ha='center', va='center', color='white' if Notak_matrizea_poly_handia[i, j] < 0.5 else "black")
plt.show()



















# with open('Modeloen_notak.txt', 'r') as informazioa:
#     info = informazioa.read()
#     print(info)