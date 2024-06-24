import numpy as np
import matplotlib.pyplot as plt
import random

def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

def stochastic(x, y, gradient, n_accuracy):
    vectors = []
    a, b = gradient(x, y)
    vectors.append(np.array([a, b]))
    for _ in range(n_accuracy):
        a_hat, b_hat = np.random.normal(a, 5), np.random.normal(b, 5)
        vectors.append(np.array([a_hat, -b_hat]))
        vectors.append(np.array([-a_hat, b_hat]))
        vectors.append(np.array([1/2*a, 2*b]))
        vectors.append(np.array([2*a, 1/2*b]))
        vectors.append(np.array([1/2*a, 1/2*b]))
    return vectors

def sgd(gradient, start, lamb=0.01, epsilon=10**(-4), random_state=20, n_accuracy=100):
    random.seed(random_state)
    np.random.seed(random_state)
    x, y = [start[0]], [start[1]]
    t = 1

    x_act, y_act = x[-1], y[-1]
    stochastic_gradients = stochastic(x_act, y_act, gradient, n_accuracy)
    grad = random.choice(stochastic_gradients)
    x.append(x_act - 1/(lamb * t) * grad[0])
    y.append(y_act - 1/(lamb * t) * grad[1])
    t += 1
    while np.sqrt((x[-1] - x[-2])**2 + (y[-1] - y[-2])**2) > epsilon:
        x_act, y_act = x[-1], y[-1]
        stochastic_gradients = stochastic(x_act, y_act, gradient, n_accuracy)
        grad = random.choice(stochastic_gradients)
        x.append(x_act - 1/(lamb * np.sqrt(t)) * grad[0])
        y.append(y_act - 1/(lamb * np.sqrt(t)) * grad[1])
        t += 1
    return x, y

def gd(gradient, start, lamb=0.01, epsilon=10**(-4), random_state=20):
    random.seed(random_state)
    np.random.seed(random_state)
    x, y = [start[0]], [start[1]]
    t = 1

    x_act, y_act = x[-1], y[-1]
    grad = gradient(x_act, y_act)
    x.append(x_act - 1/(lamb * t) * grad[0])
    y.append(y_act - 1/(lamb * t) * grad[1])
    t += 1
    while np.sqrt((x[-1] - x[-2])**2 + (y[-1] - y[-2])**2) > epsilon:
        x_act, y_act = x[-1], y[-1]
        grad = gradient(x_act, y_act)
        x.append(x_act - 1/(lamb * t) * grad[0])
        y.append(y_act - 1/(lamb * t) * grad[1])
        t += 1
    return x, y

# Parámetros iniciales y ejecución del SGD y GD
start = (5, 5)  # Inicialización en el punto (5, 5)

x, y = sgd(grad_f, start, lamb=4, epsilon=10**(-2), random_state=27, n_accuracy=50)
x_, y_ = gd(grad_f, start, lamb=3, epsilon=10**(-2), random_state=28)

# Crear un grid para los contornos
x_vals = np.linspace(-3.5, 7.5, 400)
y_vals = np.linspace(-3.5, 7.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Visualización en 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
ax.plot(x, y, [f(xi, yi) for xi, yi in zip(x, y)], color='red', marker='x', markersize=5, label="Gradientearen Beherapen Estokastikoa", zorder=5)
ax.plot(x_, y_, [f(xi_, yi_) for xi_, yi_ in zip(x_, y_)], color='blue', marker='x', markersize=5, label="Gradientearen Beherapena", zorder=5)
ax.scatter([x[0], x[-1]], [y[0], y[-1]], [f(xi, yi) for xi, yi in zip([x[0], x[-1]], [y[0], y[-1]])], color="green", s=100, zorder=10)
ax.scatter([x_[0], x_[-1]], [y_[0], y_[-1]], [f(xi_, yi_) for xi_, yi_ in zip([x_[0], x_[-1]], [y_[0], y_[-1]])], color="green", s=100, zorder=10)
fig.suptitle(r'$f(x,y) = x^2 + y^2$ Minimizazioa', x=0.8, y=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
legend = ax.legend(facecolor='white', framealpha=1)
legend.set_bbox_to_anchor((1, 0.7))
plt.show()
