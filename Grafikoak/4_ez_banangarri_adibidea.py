import random
import math
import matplotlib.pyplot as plt
import numpy as np

random.seed(122)

n1 = 30 # r = 2 zirkunferentzia barruko puntu kopurua
n2 = 40 # r = 2 zirkunferentzia kanpoko puntu kopurua
s = 100 # puntuen tamaina

x1 = []
x2 = []
erro_2 = math.sqrt(2)
# Zirkunferentzia barruko puntuak
while len(x1) < n1:
    x = random.uniform(-2,2)
    if abs(x) <= 2:
        x1.append(x)

# Zirkunferentzia kanpokoak
while len(x2) < n2:
    x = random.uniform(-5,5)
    if abs(x) > 2:
        x2.append(x)
plt.scatter(x1,np.zeros((1,len(x1))), s = s, color = "green", marker= "+",label = "")
plt.scatter(x2,np.zeros((1,len(x2))),s = s, color = "red", marker= "_",label = "")
plt.xlim([-5,5])
plt.ylim([-.2 ,.2])
# plt.grid()
# plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel(r'$x$', fontsize = 17)
# l = plt.legend(["+","-"])
# l.get_frame().set_alpha(1) 
plt.tight_layout(pad = 0.2)
plt.text(1.3, .17, r"$\mathcal{X} = [-5,5] \subset \BbbR$", fontsize=20, color='black')
plt.show()





# 2d grafikoa
x1 = np.array(x1)
y1 = x1**2

x2 = np.array(x2)
y2 = x2**2

plt.scatter(x1,y1, s = s, color = "green", marker= "+")
plt.scatter(x2,y2,s = s, color = "red", marker= "_")
# plt.grid()
# plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel(r'$x$', fontsize = 17)
plt.ylabel(r'$x^2$', rotation = 0, fontsize = 17)
# l = plt.legend(["+","-"])
# l.get_frame().set_alpha(1) 


m = 27 / 130 - 0.05
b = 12227 / 3250
# Lerroa marrazteko puntuak:
x = np.linspace(-5, 5, 100)
y = m * x + b
plt.plot(x,y,linewidth = 3)
# l = plt.legend(["+","-"], loc = "upper center")
# l.get_frame().set_alpha(1) 
plt.xlim([-4,4])
plt.ylim([-2,16])
plt.text(2.4, 14.5, r"$\hat{\mathcal{X}} \subset \BbbR^2$", fontsize=20, color='black')
plt.tight_layout(pad = 0.2)
plt.show()

