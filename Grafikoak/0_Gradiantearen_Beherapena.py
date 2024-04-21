import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir la función z en términos de x y y
def f(x1,x2):
    return 1.25 *(x1 + 5)**2 + (x2-8)**2

def gradiant_f(x1,x2):
    return np.array([2*1.25*(x1+6), 2*(x2-8)])


def GD(f,grad_f,x0,iter = 100, mu = 0.1):
    W = np.zeros((iter, 2))
    W[0,:] = x0
    for t in range(iter-1):
        W[t+1,:] = W[t,:] - mu * grad_f(W[t,0],W[t,1])
    return 1/iter * sum(W), W

minim, W = GD(f= f, grad_f= gradiant_f, x0= np.array([-30,40]), iter = 100, mu = 0.1)


# Generar datos para x, y
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
x, y = np.meshgrid(x, y)

# Calcular los valores de z
z = f(x, y)

# Crear la figura
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Graficar el contorno de la función

altuera = np.array([1000 for i in range(100)])
contorno = ax.plot_surface(x, y, z, cmap = plt.cm.cividis, alpha = 1)
puntos = ax.scatter(W[:,0], W[:,1], f(W[:,0],W[:,1]) + altuera, c = 'r', s = 10,alpha = 1)

# Etiquetas de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Mostrar el gráfico
plt.show()

