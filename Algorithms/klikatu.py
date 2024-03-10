import numpy as np
import matplotlib.pyplot as plt
from SGD_soft_SVM import soft_SVM_SGD


lim = 10

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
print(x_vectors)
print(y_vectors)

# ALGORITMOA EXEKUTATU
w_hat,theta,w,x_berriak,mean,sd = soft_SVM_SGD(x_vectors,y_vectors,10000,1,standardize=True,plot=True,lim = lim)
b = w[0]
b2 = w_hat[-1,0]

x = np.linspace(-lim, lim, 100)
y = -b / w[2] - w[1]/w[2]*x
#y2 = -b2 / w_hat[-1,2] - w_hat[-1,1]/w_hat[-1,2]*x
y_originala = mean[1] + sd[1]/w[2] * (-b - w[1]/sd[0]*(x-mean[0]))

plt.plot(x,y)
plt.scatter(x_berriak[:,1],x_berriak[:,2],c = y_vectors, cmap = "viridis")
plt.grid()
plt.title("SVM-leuna SGD algoritmoarekin estandarizatua")
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.show()


plt.plot(x,y_originala)
plt.scatter(x_vectors[:,0],x_vectors[:,1],c = y_vectors, cmap = "viridis")
plt.grid()
plt.title("SVM-leuna SGD algoritmoarekin originala")
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
plt.show()




print(f"w_hat = {w_hat}")
print(w)

print(f"theta = {theta}")