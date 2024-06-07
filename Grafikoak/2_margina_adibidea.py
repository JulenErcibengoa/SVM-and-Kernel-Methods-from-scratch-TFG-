import matplotlib.pyplot as plt
import numpy as np
import random

def sortu_puntuak (n_puntu , x_min, x_max, y_min, y_max):
    kx = []
    ky = []
    while len(kx) != n_puntu:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        kx.append(x)
        ky.append(y)
    return kx, ky

# Lerroa marrazteko puntuak:
x = np.linspace(-10, 10, 100)
y1 = x - 0.7
y2 = x/3 + 0.5

y3 = -0.4 + x - 1
y4 = x

y5 = -0.4 + 1/3*(x-1)
y6 = 1.81999 + 1/3*(x-0.26)
# Grafikoa sortu: 


plt.plot(x, y3, label = 'Margin txikia', linestyle = "dotted", alpha = 0.7, c = "tab:blue")
plt.plot(x, y4, label = 'Margin txikia', linestyle = "dotted", alpha = 0.7, c = "tab:blue")
plt.plot(x, y5, label = 'Margin txikia', linestyle = "dotted", alpha = 0.7, c = "tab:orange")
plt.plot(x, y6, label = 'Margin txikia', linestyle = "dotted", alpha = 0.7, c = "tab:orange")



hiper_1, = plt.plot(x, y1, label='Margin txikia', linewidth = 2.2, linestyle = "--")
hiper_2, = plt.plot(x, y2, label = 'Margin handia', linewidth = 2.2)
plt.xlabel(r'$x_1$', fontsize = 17)
plt.ylabel(r'$x_2$',rotation = 0, fontsize = 17)
plt.tight_layout(pad = 0.2)

# plt.grid(True)
plt.xlim([-3.2,3.2])
plt.ylim([-3.2,3.2])
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')

# Puntuak gehitu
random.seed(121)

n = 15
# Negatiboak
kx_minus, ky_minus = sortu_puntuak(n+5, 0.6, 3.5, -0.6, -3.5) 
plt.scatter(kx_minus, ky_minus, color='red', marker='_', s = 300)
kx_minus, ky_minus = sortu_puntuak(n-n//2, 1.8, 3.5, -0.3, -1) 
plt.scatter(kx_minus, ky_minus, color='red', marker='_', s = 300)

n = 20
kx_plus, ky_plus = sortu_puntuak(n, -1.4, -3.5, 1.4, 3.5) 
plt.scatter(kx_plus, ky_plus, color='green', marker='+', s = 300)
kx_minus, ky_minus = sortu_puntuak(n-n//3 * 2, -0.3, -1,2.2, 3.5) 
plt.scatter(kx_minus, ky_minus, color='green', marker='+', s = 300)


# Distantzien adierazpenak:
plt.plot([1, 13/20], [-0.4, -0.05], color='black',linewidth = 2)
plt.plot([1, 63/100], [-0.4, 71/100], color='black',linewidth = 2)
plt.scatter([1],[-0.4], color='red', marker='_', s = 500, linewidth = 4)

l = plt.legend(handles=[hiper_1, hiper_2], fontsize=15, loc='center left', bbox_to_anchor=(1, 0.5))
l.get_frame().set_alpha(1)

plt.show()
