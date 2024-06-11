import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

def twoD_GD(f, df, x0, iter = 100, mu = 0.1):
    W = np.zeros(iter)
    W[0] = x0
    for t in range(iter-1):
        W[t+1] = W[t]- mu * df(W[t])
    return 1/iter * sum(W), W


x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
x, y = np.meshgrid(x, y)


z = f(x, y)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')



altuera = np.array([10 for i in range(100)])
contorno = ax.plot_surface(x, y, z, cmap = plt.cm.cividis, alpha = 0.5)
puntos = ax.scatter(W[:,0], W[:,1], f(W[:,0],W[:,1]) + altuera, c = 'r', s = 10,alpha = 1)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


plt.show()





# 2D

x = np.linspace(-4,4,100)

def funct(x):
    return x**2

def df(x):
    return 2*x

y = funct(x)

def gd(f,df,x0,nu,T):
    v = np.zeros(T+1)
    v[0] = x0
    for i in range(1, T+1):
        print(i)
        v[i] = v[i-1] - nu * df(v[i-1])
    return v, v[-1]

v, minim = gd(funct, df, -4, 0.1, 30)
print(v)
print(minim)

plt.plot(x,y)
plt.scatter(v, funct(v),color = "red")
plt.show()
        




