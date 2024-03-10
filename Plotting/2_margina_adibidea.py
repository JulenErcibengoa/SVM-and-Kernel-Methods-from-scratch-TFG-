import matplotlib.pyplot as plt
import numpy as np



# Lerroa marrazteko puntuak:
x = np.linspace(-10, 10, 100)
y1 = 0.85*x + 0.9
y2 = 0.55 *x + 0.65

# Grafikoa sortu: 
plt.plot(x, y1, label='Margin handia')
plt.plot(x, y2, label = 'Margin txikia')
plt.xlabel('X')
plt.ylabel('Y')

plt.grid(True)
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')

# Puntuak gehitu
plt.scatter([2.3, -0.4, -1.4,0.6,1.5], [1.2, -2.3,-1.2,-0.4,-1.5], color='red', marker='.', s = 300)
plt.scatter([-1.7, -0.4, 0.5,-2.5,-2.6], [0.7, 1.4, 2.6,1.4,-0.5], color='green', marker='.', s = 300)
plt.scatter([1.1],[1.1], color='red', marker='.', s = 300)

l =plt.legend(fontsize = 12, loc = 'upper left')
l.get_frame().set_alpha(1) 

plt.show()
