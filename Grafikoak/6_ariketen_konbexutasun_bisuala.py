import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-6, 6, 100)

y = 2 - x

y1 = 1 - x

y2 = 3 - x
plt.axis('equal') 


plt.vlines(x = 0, ymin = -2, ymax = 4, colors = "black")
plt.hlines(y = 0, xmin = -2, xmax = 4, colors = "black")
# plt.grid()

plt.plot(x,y1, linestyle = "--", label =  r"$x_1 + x_2 - 1 = 0$")

plt.plot(x,y,label = r"$x_1 + x_2 - 2 = 0$")

plt.plot(x,y2, linestyle = "dotted", label =  r"$x_1 + x_2 - 3 = 0$")

plt.xlim([-1,3])
plt.ylim([-1,3])

plt.scatter([1],[0.5], c="green", s = 400, marker= "+")
plt.quiver(0,0,1,1, angles='xy', scale_units='xy', scale=1)

plt.xlabel(r"$x_1$",fontsize = 20)

plt.ylabel(r"$x_2$",fontsize = 20, rotation = 0)
plt.text(.5,.8, r"$w$",fontsize = 20)




plt.legend()
plt.show()
