import numpy as np
import matplotlib.pyplot as plt
from SGD_soft_SVM import soft_SVM_SGD
from SGD_soft_SVM_Kernels import Nire_SGD_kernelekin,gaussian_kernel,polynomial_kernel


# # Adibidea

# Lagina
X = np.array([[2.3,1.2],[-1.7,0.7],[-0.4,-2.3],[-0.4,1.4],[-1.4,-1.2],[0.5,2.6],[0.6,-0.4],[-2.5,1.4],[1.5,-1.5],[-2.6,-0.5],[3,3],[3.2,3.1],[2.8,3.5],[3.5,3.1],[3,4.2],[3.8,4],[-4,4],[-4.6,3.8],[-4.2,3.9],[-4,4.8],[-4.4,3.9]])
Y = np.array([1,-1,1,-1,1,-1,1,-1,1,-1,2,2,2,2,2,2,3,3,3,3,3])

# Emaitza
model = Nire_SGD_kernelekin(1,"kernel polinomiala")
model.fit(X,Y,100000)
print("Fitted!")

# Plot
x_plot = np.linspace(-5,5,200)
y_plot = np.linspace(-5,5,200)

pos = []
neg = []
bestea = []
hurrengoak = []

for x in x_plot:
    for y in y_plot:
        predikzioa = model.predict([x,y])
        if  predikzioa == 1:
            pos.append([x,y])
        elif predikzioa == -1:
            neg.append([x,y])
        elif predikzioa == 2:
            bestea.append([x,y])
        elif predikzioa == 3:
            hurrengoak.append([x,y])    

pos = np.array(pos)
neg = np.array(neg)
bestea = np.array(bestea)
hurrengoak = np.array(hurrengoak)



plt.scatter(pos[:,0],pos[:,1],c = "green",alpha = 0.5)
if len(neg) > 0:
    plt.scatter(neg[:,0],neg[:,1],c="red",alpha = 0.5)
plt.scatter(bestea[:,0],bestea[:,1],c="orange",alpha = 0.5)
plt.scatter(hurrengoak[:,0],hurrengoak[:,1],c="blue",alpha = 0.5)

plt.scatter(X[:,0],X[:,1],c = Y,cmap="viridis")

plt.xlim([-5,5])
plt.ylim([-5,5])

plt.title(f"Modeloaren zehaztasuna = {model.score(X,Y)}")
plt.show()













lim = 3

# Listas para almacenar las coordenadas
coordenadas_x_pos = []
coordenadas_y_pos = []

def click_pos(event):
    # Al hacer clic, se añaden las coordenadas a las listas
    coordenadas_x_pos.append(event.xdata)
    coordenadas_y_pos.append(event.ydata)
    # Se actualiza el gráfico con el nuevo punto
    plt.scatter(event.xdata, event.ydata, c='green')
    plt.title(f"Puntu positibo kopurura: {len(coordenadas_x_pos)}")
    plt.draw()
# Crear un gráfico vacío
fig, ax = plt.subplots()
ax.set_title('Puntu positiboak gehitu')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
# Conectar el evento de clic con la función onclick
fig.canvas.mpl_connect('button_press_event', click_pos)
# Mostrar el gráfico
plt.show()


# Listas para almacenar las coordenadas
coordenadas_x_neg = []
coordenadas_y_neg = []

def click_neg(event):
    # Al hacer clic, se añaden las coordenadas a las listas
    coordenadas_x_neg.append(event.xdata)
    coordenadas_y_neg.append(event.ydata)
    # Se actualiza el gráfico con el nuevo punto
    plt.scatter(event.xdata, event.ydata, c='red')
    plt.title(f"Punto positibo kopurua: {len(coordenadas_x_pos)}\nPuntu negatibo kopurura: {len(coordenadas_x_neg)}")
    plt.draw()
# Crear un gráfico vacío
fig, ax = plt.subplots()
ax.set_title('Puntu negatiboak gehitu')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
plt.scatter(coordenadas_x_pos,coordenadas_y_pos,c = "green")
# Conectar el evento de clic con la función onclick
fig.canvas.mpl_connect('button_press_event', click_neg)
# Mostrar el gráfico
plt.show()


# Datu-basea sortu:
x_coord = np.concatenate(( np.array(coordenadas_x_pos), np.array(coordenadas_x_neg)))
y_coord = np.concatenate(( np.array(coordenadas_y_pos), np.array(coordenadas_y_neg)))

x_vectors = np.column_stack((x_coord,y_coord)) # Algoritmoan sartzeko prest
y_vectors = np.concatenate((np.array([1 for x in coordenadas_x_pos]),np.array([-1 for x in coordenadas_x_neg])))













# ALGORITMOA (KERNEL)

modeloa = Nire_SGD_kernelekin(0.01,"kernel gaussiarra")
modeloa.fit(x_vectors,y_vectors)
# Plot

n = 200

x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
m = len(x_vectors)

X,Y = np.meshgrid(x,y)
Z = np.zeros([n,n])



pos_x = []
pos_y = []

neg_x = []
neg_y = []

x_new = np.concatenate( (np.ones((m,1)),x_vectors) , axis = 1)

for i in range(n):
    for j in range(n):  
        if modeloa.predict([x[i],y[j]]) == 1:
            Z[j,i] = 1
            pos_x.append(i)
            pos_y.append(j)
        else:
            Z[j,i] = -1
            neg_x.append(i)
            neg_y.append(j)

print(Z)
#plt.scatter(pos_x,pos_y,c = "green",alpha = 0.5)
#plt.scatter(neg_x,neg_y,c="red",alpha = 0.5)

plt.contourf(X, Y, Z, cmap='viridis') 

plt.scatter(x_vectors[:,0],x_vectors[:,1],c = y_vectors,cmap="viridis", edgecolors= "black")
plt.title(f"Modeloaren zehaztasuna = {modeloa.score(x_vectors,y_vectors)}")

plt.show()





























# # ALGORITMOA EXEKUTATU
# w_hat,theta,w,x_berriak,mean,sd = soft_SVM_SGD(x_vectors,y_vectors,10000,1,standardize=True,plot=True,lim = lim)
# b = w[0]

# x = np.linspace(-lim, lim, 100)
# y = -b / w[2] - w[1]/w[2]*x
# y_originala = mean[1] + sd[1]/w[2] * (-b - w[1]/sd[0]*(x-mean[0]))


# a,b,w_dis_gabe,k,k,k = soft_SVM_SGD(x_vectors,y_vectors,10000,1,standardize=False)
# b_dis_gabe = w_dis_gabe[0]
# y_dis_gabe = -b_dis_gabe / w_dis_gabe[2] - w_dis_gabe[1]/w_dis_gabe[2]*x
# # grafikoak

# fig, axs = plt.subplots(1, 3, figsize=(15, 4))
# axs[0].plot(x,y)
# axs[0].scatter(x_berriak[:,1],x_berriak[:,2],c = y_vectors, cmap = "viridis")
# axs[0].set_title('Algoritmo estandarizatua\nDatu estandarizatuak')
# axs[0].grid()
# axs[0].set_xlim([-3,3])
# axs[0].set_ylim([-3,3])

# axs[1].plot(x,y_originala)
# axs[1].scatter(x_vectors[:,0],x_vectors[:,1],c = y_vectors, cmap = "viridis")
# axs[1].set_title('Algoritmo estandarizatua\nDatu originalak')
# axs[1].grid()
# axs[1].set_xlim([-lim,lim])
# axs[1].set_ylim([-lim,lim])

# axs[2].plot(x,y_dis_gabe)
# axs[2].scatter(x_vectors[:,0],x_vectors[:,1],c = y_vectors, cmap = "viridis")
# axs[2].set_title('Algoritmo estandarizatu gabe\nDatu originalak')
# axs[2].grid()
# axs[2].set_xlim([-lim,lim])
# axs[2].set_ylim([-lim,lim])
# plt.show()




