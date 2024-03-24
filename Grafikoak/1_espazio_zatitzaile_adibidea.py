import matplotlib.pyplot as plt
import numpy as np

m = 1
b = -1
# Lerroa marrazteko puntuak:
x = np.linspace(-10, 10, 100)
y = m * x - b

# Grafikoa sortu: 
plt.plot(x, y, label=r'$-x + y - 1 = 0$')
plt.title('Espazio zatitzailea '+r'$\vec{w} = (-1,1)$' + ' eta ' + r'$b = -1$',fontsize = 15)
plt.xlabel('X')
plt.ylabel('Y')

plt.grid(True)
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')

# Puntuak gehitu
plt.scatter([2.3, -0.4, -1.4,0.6,1.5], [1.2, -2.3,-1.2,-0.4,-1.5], color='red', marker='_', s = 300)
plt.scatter([-1.7, -0.4, 0.5,-2.5,-2.6], [0.7, 1.4, 2.6,1.4,-0.5], color='green', marker='+', s = 300)
plt.scatter([1],[1], color='red', marker='o', s = 100)
plt.text(1.1,0.7, r'$(1,1)$', color='black', fontsize=12)

# Bektorea gehitu
q = plt.quiver(0,0,-1,1,angles = 'xy',scale_units = 'xy', scale = 1, color = 'black')
plt.text(-1.3,1.1, r'$\vec{w}$', color='black', fontsize=12)

l =plt.legend(fontsize = 15, loc = 'upper left')
l.get_frame().set_alpha(1) 
plt.show()

