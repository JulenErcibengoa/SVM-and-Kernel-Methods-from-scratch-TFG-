import numpy as np
import matplotlib.pyplot as plt

# Define tu función f
def f(x, y):
    return x**2 + y**2  # Puedes reemplazar esto con tu propia función

# Define el rango de valores para x y y
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

# Crea una malla de valores para x e y
X, Y = np.meshgrid(x, y)

# Calcula los valores de la función f en cada posición de la malla
Z = f(X, Y)
print(np.shape(X))
print(np.shape(Y))
print(np.shape(Z))



# Crea el gráfico de contorno con el fondo basado en la función f
plt.contourf(X, Y, Z, cmap='viridis')  # Puedes cambiar 'viridis' por otro mapa de colores

# Agrega un contorno de líneas para visualizar mejor la función
plt.contour(X, Y, Z, colors='black', linestyles='dashed', linewidths=0.5)

# Añade etiquetas y título
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico con fondo basado en la función f')

# Muestra el gráfico
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


points = np.ones(5)  # Draw 5 points for each line
marker_style = dict(color='tab:blue', linestyle=':', marker='o',
                    markersize=15, markerfacecoloralt='tab:red')

fig, ax = plt.subplots()

# Plot all fill styles.
for y, fill_style in enumerate(Line2D.fillStyles):
    ax.text(-0.5, y, repr(fill_style),
            horizontalalignment='center', verticalalignment='center')
    ax.plot(y * points, fillstyle=fill_style, **marker_style)

ax.set_axis_off()
ax.set_title('fill style')

plt.show()



M = np.zeros((5,5))
k = 0
for i in range(1,5):
    for j in range(i):
        k += 1
        M[i,j] = k


print(M)