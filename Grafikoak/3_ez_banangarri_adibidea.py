import random
import math
import matplotlib.pyplot as plt
import numpy as np

random.seed(124)

n1 = 70 # r = 2 zirkunferentzia barruko puntu kopurua
n2 = 100 # r = 2 zirkunferentzia kanpoko puntu kopurua
s = 50 # puntuen tamaina

x1 = []
y1 = []
x2 = []
y2 = []
erro_2 = math.sqrt(2)
# Zirkunferentzia barruko puntuak
while len(x1) < n1:
    x = random.uniform(-2,2)
    y = random.uniform(-2,2)
    if x**2 + y**2 <= erro_2+0.4:
        x1.append(x)
        y1.append(y)

# Zirkunferentzia kanpokoak
while len(x2) < n2:
    x = random.uniform(-3,3)
    y = random.uniform(-3,3)
    if (x**2 + y**2 > (erro_2 - 0.4)) and (x**2 + y**2 <= 4):
        x2.append(x)
        y2.append(y)
plt.scatter(x1,y1, s = s, color = "green", marker= "+")
plt.scatter(x2,y2,s = s, color = "red", marker= "_")
plt.xlim([-2,2])
plt.ylim([-2,2])
# plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
# plt.title("Datu ez-linealak",fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 17)
plt.ylabel(r'$x_2$', rotation = 0, fontsize = 17)
# l = plt.legend(["+","-"])
# l.get_frame().set_alpha(1) 
plt.tight_layout(pad = .2)
plt.show()

# 3d grafikoa
x1 = np.array(x1)
y1 = np.array(y1)
z1 = (x1**2 + y1**2)

x2 = np.array(x2)
y2 = np.array(y2)
z2 = (x2**2 + y2**2)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(x1,y1,z1, alpha = 1,color = "green")
ax.scatter(x2,y2,z2,alpha = 1,color = "red")


# Planoa grafikatu
def z_sqrt2(x,y):
    return np.sqrt(2) * x / x
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
x, y = np.meshgrid(x, y)
z = z_sqrt2(x, y)

ax.plot_surface(x, y, z, alpha=0.8, color='c')

plt.show()