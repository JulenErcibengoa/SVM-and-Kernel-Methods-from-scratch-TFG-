import random
import math
import matplotlib.pyplot as plt
import numpy as np

# 1. grafikoa

# Puntuak gehitu
plt.scatter([2.3, -0.4, -1.4,0.6,1.5], [1.2, -2.3,-1.2,-0.4,-1.5], color='red', marker='_', s = 300)
plt.scatter([-0.4, 0.8,-2.5,-2.6, -1.4], [1.9, 2.6,1.4,-0.5, 2.7], color='green', marker='+', s = 300)
plt.scatter([1],[1], color='red', marker='_', s = 500, linewidth = 4)
plt.scatter([-1],[1], color='green', marker='+', s = 500, linewidth = 4)
plt.text(0.5,0.5, r'$(1,1)$', color='black', fontsize=20)
plt.text(-1.8,0.5, r'$(-1,1)$', color='black', fontsize=20)
plt.xlabel(r'$x_1$', fontsize = 17)
plt.ylabel(r'$x_2$',rotation = 0, fontsize = 17)
plt.tight_layout(pad = 0.2)

# plt.grid(True)
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')



# l =plt.legend(fontsize = 15, loc = 'upper left')
# l.get_frame().set_alpha(1) 
plt.show()




# 2. grafikoa

m = 1
b = -0.8
# Lerroa marrazteko puntuak:
x = np.linspace(-10, 10, 100)
y = m * x - b

# Grafikoa sortu: 
plt.plot(x, y, label=r'$-x + y - 1 = 0$')
# Puntuak gehitu
plt.scatter([2.3, -0.4, -1.4,0.6,1.5], [1.2, -2.3,-1.2,-0.4,-1.5], color='red', marker='_', s = 300)
plt.scatter([ 0.8,-2.5,-2.6, -1.4], [2.6,1.4,-0.5, 2.7], color='green', marker='+', s = 300)
plt.scatter([1],[1], color='red', marker='_', s = 500, linewidth = 4)
plt.scatter([-1],[1], color='green', marker='+', s = 500, linewidth = 4)
plt.text(0.5,0.5, r'$(1,1)$', color='black', fontsize=20)
plt.text(-1.8,0.5, r'$(-1,1)$', color='black', fontsize=20)
plt.xlabel(r'$x_1$', fontsize = 17)
plt.ylabel(r'$x_2$',rotation = 0, fontsize = 17)
plt.tight_layout(pad = 0.2)

# plt.grid(True)
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')

q = plt.quiver(.5,1.3,-1,1,angles = 'xy',scale_units = 'xy', scale = 1, color = 'black')
plt.text(-1,2.3, r'$w$', color='black', fontsize=20)


# l =plt.legend(fontsize = 15, loc = 'upper left')
# l.get_frame().set_alpha(1) 
plt.show()
