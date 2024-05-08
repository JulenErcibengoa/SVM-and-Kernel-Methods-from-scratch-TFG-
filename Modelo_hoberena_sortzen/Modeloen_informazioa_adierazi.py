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
from sklearn.metrics import confusion_matrix

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
random.seed(123)
fig, axs = plt.subplots(2,5, figsize = (12,8))
k = random.randint(0, len(Y_test)-1)
for i in range(2):
    for j in range(5):
        predikzioa =  Nire_gauss.predict(X_test.iloc[k,:])
        zuzena = Y_test.iloc[k]
        while predikzioa == zuzena:
            k = random.randint(0, len(Y_test)-1)
            predikzioa =  Nire_gauss.predict(X_test.iloc[k,:])
            zuzena = Y_test.iloc[k]
        adib = np.array(X_test.iloc[k, :])
        adib = np.reshape(adib, (28,28))
        axs[i,j].imshow(adib, cmap = "gray_r")
        axs[i,j].axis("off")
        axs[i,j].text(0.5,1.2,f"Predikzioa =  {str(predikzioa)}\nZuzena = {str(zuzena)}", fontsize = 15, horizontalalignment = "center", verticalalignment = "top", transform = axs[i,j].transAxes)
        k += 1

plt.tight_layout(pad = 0)
# plt.show()
plt.savefig(os.path.join(oraingo_bidea, "Irudiak", "Nire_gauss_gaizki_klasifikatuak.png"))




# # # ----------------------------------------------------------------------------
# # # --------------------NIRE MODELOA KERNEL POLINOMIALA-------------------------
# # # ----------------------------------------------------------------------------

# # # Modeloa inportatu
# Nire_poly = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_poly.pkl"),"rb"))
# Nire_poly_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_poly_info.pkl"),"rb"))

# # # Informazioa erakutsi:

# print("Nire modeloa, kernel polinomiala:")
# print(f"Behar izan duen denbora: {Nire_poly_info[0]}")
# print(f"Lortu duen asmatze-proportzioa: {Nire_poly_info[1]}")
# print()




# # # Gaizki egindako predikzioak bisualizatu:
# fig, axs = plt.subplots(2,5, figsize = (12,8))
# k = random.randint(0, len(Y_test)-1)
# for i in range(2):
#     for j in range(5):
#         predikzioa =  Nire_poly.predict(X_test.iloc[k,:])
#         zuzena = Y_test.iloc[k]
#         while predikzioa == zuzena:
#             k = random.randint(0, len(Y_test)-1)
#             predikzioa =  Nire_poly.predict(X_test.iloc[k,:])
#             zuzena = Y_test.iloc[k]
#         adib = np.array(X_test.iloc[k, :])
#         adib = np.reshape(adib, (28,28))
#         axs[i,j].imshow(adib, cmap = "gray_r")
#         axs[i,j].axis("off")
#         axs[i,j].text(0.5,1.2,f"Predikzioa =  {str(predikzioa)}\nZuzena = {str(zuzena)}", fontsize = 15, horizontalalignment = "center", verticalalignment = "top", transform = axs[i,j].transAxes)
#         k += 1

# plt.tight_layout(pad = 0)
# # plt.show()
# plt.savefig(os.path.join(oraingo_bidea, "Irudiak", "Nire_poly_gaizki_klasifikatuak.png"))




# # ----------------------------------------------------------------------------
# # --------------------SCIKIT MODELOA KERNEL GAUSSIARRA--------------------------
# # ----------------------------------------------------------------------------

# # Modeloa inportatu
Scikit_gauss = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_gauss.pkl"),"rb"))
Scikit_gauss_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_gauss_info.pkl"),"rb"))

# # Informazioa erakutsi:
print("Nire modeloa, kernel gausiarra:")
print(f"Behar izan duen denbora: {Scikit_gauss_info[0]}")
print(f"Lortu duen asmatze-proportzioa: {Scikit_gauss_info[1]}")
print()

# # Conffusion Matrix egin:

# predikzioak_Scikit_gauss = Scikit_gauss.predict(X_test)
# conf_matrix_Scikit_gauss = confusion_matrix(Y_test, predikzioak_Scikit_gauss, labels= np.unique(Y_test.values))
# pickle.dump(conf_matrix_Scikit_gauss, open(os.path.join(oraingo_bidea, "Conf_matrix_gordeta", "conf_matrix_Scikit_gauss"),"wb"))
conf_matrix_Scikit_gauss = pickle.load(open(os.path.join(oraingo_bidea, "Conf_matrix_gordeta", "conf_matrix_Scikit_gauss"),"rb"))

matrizea = conf_matrix_Scikit_gauss
plt.figure(figsize=(8, 7))
plt.imshow(matrizea, aspect= "equal")
plt.xticks(np.arange(0, 10, 1), [r'${{{}}}$'.format(j) for j in range(10)], fontsize = 17)
plt.xlabel("Predikzioa", fontsize = 20)
plt.yticks(np.arange(0, 10, 1), [r'${{{}}}$'.format(j) for j in range(10)], fontsize = 17)
plt.ylabel("Benetako balioa", fontsize = 20)
# plt.title("Modelo desberdinen asmatze proportzioa baliozta-multzoan:\nKernel gaussiarra")
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=10)
plt.clim(0,1200)
plt.tight_layout(pad = 0.2)
for i in range(matrizea.shape[0]):
    for j in range(matrizea.shape[1]):
        plt.text(j, i, '{:.0f}'.format(matrizea[i, j]), ha='center', va='center', color='white' if matrizea[i, j] < 500 else "black", fontsize = 15)
plt.savefig(os.path.join(oraingo_bidea, "Irudiak", "Scikit_gauss_conf_matrix.png"))



# # # Gaizki egindako predikzioak bisualizatu:
# fig, axs = plt.subplots(2,5, figsize = (12,8))
# k = random.randint(0, len(Y_test)-1)
# for i in range(2):
#     for j in range(5):
#         predikzioa =  Scikit_gauss.predict(np.array(X_test.iloc[k,:].tolist()).reshape(1,-1))[0]
#         zuzena = Y_test.iloc[k]
#         while predikzioa == zuzena:
#             k = random.randint(0, len(Y_test)-1)
#             predikzioa =  Scikit_gauss.predict(np.array(X_test.iloc[k,:].tolist()).reshape(1,-1))[0]
#             zuzena = Y_test.iloc[k]
#         adib = np.array(X_test.iloc[k, :])
#         adib = np.reshape(adib, (28,28))
#         axs[i,j].imshow(adib, cmap = "gray_r")
#         axs[i,j].axis("off")
#         axs[i,j].text(0.5,1.2,f"Predikzioa =  {str(predikzioa)}\nZuzena = {str(zuzena)}", fontsize = 15, horizontalalignment = "center", verticalalignment = "top", transform = axs[i,j].transAxes)
#         k += 1

# plt.tight_layout(pad = 0)
# # plt.show()
# plt.savefig(os.path.join(oraingo_bidea, "Irudiak", "Scikit_gauss_gaizki_klasifikatuak.png"))





# # # ----------------------------------------------------------------------------
# # # --------------------SCIKIT MODELOA KERNEL POLINOMIALA-------------------------
# # # ----------------------------------------------------------------------------

# # Modeloa inportatu
Scikit_poly = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_poly.pkl"),"rb"))
Scikit_poly_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_poly_info.pkl"),"rb"))

# # Informazioa erakutsi:

print("Nire modeloa, kernel polinomiala:")
print(f"Behar izan duen denbora: {Scikit_poly_info[0]}")
print(f"Lortu duen asmatze-proportzioa: {Scikit_poly_info[1]}")
print()


# # Conffusion Matrix egin:

# predikzioak_Scikit_poly = Scikit_poly.predict(X_test)
# conf_matrix_Scikit_poly = confusion_matrix(Y_test, predikzioak_Scikit_poly, labels= np.unique(Y_test.values))
# pickle.dump(conf_matrix_Scikit_poly, open(os.path.join(oraingo_bidea, "Conf_matrix_gordeta", "conf_matrix_Scikit_poly"),"wb"))
conf_matrix_Scikit_poly = pickle.load(open(os.path.join(oraingo_bidea, "Conf_matrix_gordeta", "conf_matrix_Scikit_poly"),"rb"))

matrizea = conf_matrix_Scikit_poly
plt.figure(figsize=(8, 7))
plt.imshow(matrizea, aspect= "equal")
plt.xticks(np.arange(0, 10, 1), [r'${{{}}}$'.format(j) for j in range(10)], fontsize = 17)
plt.xlabel("Predikzioa", fontsize = 20)
plt.yticks(np.arange(0, 10, 1), [r'${{{}}}$'.format(j) for j in range(10)], fontsize = 17)
plt.ylabel("Benetako balioa", fontsize = 20)
# plt.title("Modelo desberdinen asmatze proportzioa baliozta-multzoan:\nKernel gaussiarra")
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=10)
plt.clim(0,1200)
plt.tight_layout(pad = 0.2)
for i in range(matrizea.shape[0]):
    for j in range(matrizea.shape[1]):
        plt.text(j, i, '{:.0f}'.format(matrizea[i, j]), ha='center', va='center', color='white' if matrizea[i, j] < 500 else "black", fontsize = 15)
plt.savefig(os.path.join(oraingo_bidea, "Irudiak", "Scikit_poly_conf_matrix.png"))



# # # Gaizki egindako predikzioak bisualizatu:
# fig, axs = plt.subplots(2,5, figsize = (12,8))
# k = random.randint(0, len(Y_test)-1)
# for i in range(2):
#     for j in range(5):
#         predikzioa =  Scikit_poly.predict(np.array(X_test.values[k]).reshape(1,-1))[0]
#         zuzena = Y_test.iloc[k]
#         while predikzioa == zuzena:
#             k = random.randint(0, len(Y_test)-1)
#             predikzioa =  Scikit_poly.predict(np.array(X_test.values[k]).reshape(1,-1))[0]
#             zuzena = Y_test.iloc[k]
#         adib = np.array(X_test.iloc[k, :])
#         adib = np.reshape(adib, (28,28))
#         axs[i,j].imshow(adib, cmap = "gray_r")
#         axs[i,j].axis("off")
#         axs[i,j].text(0.5,1.2,f"Predikzioa =  {str(predikzioa)}\nZuzena = {str(zuzena)}", fontsize = 15, horizontalalignment = "center", verticalalignment = "top", transform = axs[i,j].transAxes)
#         k += 1

# plt.tight_layout(pad = 0)
# # plt.show()
# plt.savefig(os.path.join(oraingo_bidea, "Irudiak", "Scikit_poly_gaizki_klasifikatuak.png"))