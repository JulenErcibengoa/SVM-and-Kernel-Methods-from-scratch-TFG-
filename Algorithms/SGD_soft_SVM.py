import random
import numpy as np
import matplotlib.pyplot as plt

def soft_SVM_SGD(x,y,T = 100,lamb = 1,standardize = True, plot = False):
    """
    x = m luzeerako bektore multzoa (domeinua)
    y = m luzeerako bektorea (izenak)
    lamb = lambda parametroa
    T = iterazio kopurua
    """
    # Beharrezko aldagaiak sortu:
    m = len(x)
    d = len(x[0]) + 1 # d + 1 dimentsioan lan egingo dugu, hiperplano homogeneoa bilatuz: w' = (b,w1,w2...,wd) eta x' = (1,x1,x2,...,xd)
    theta = theta = np.zeros([T+1,d])
    w = np.zeros([T,d])

    if standardize: 
        # Lan egingo ditugun datuak sortu (estandarizatuak)
        mean = np.mean(x,0)
        sd = np.std(x,0)
        x_new = np.concatenate( (np.ones((m,1)),(x-mean)/sd) , axis = 1)
    else: 
        x_new =  np.concatenate( (np.ones((m,1)),x) , axis = 1)

    if plot: 
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plt.grid()
        axs[0].scatter(x[:,0],x[:,1],c = y, cmap = "viridis")
        axs[0].set_title('Datu originalak')
        axs[0].grid()
        axs[0].set_xlim([-3,3])
        axs[0].set_ylim([-3,3])
        # plt.xlim([np.min(x[:,0]),np.max(x[:,0])])
        # plt.ylim([np.min(x[:,1]),np.max(x[:,1])])
        axs[1].scatter(x_new[:,1],x_new[:,2],c = y, cmap = "viridis")
        axs[1].set_title('Datu berriak')
        axs[1].set_xlim([-3,3])
        axs[1].set_ylim([-3,3])
        # plt.xlim([np.min(x[:,0]),np.max(x[:,0])])
        # plt.ylim([np.min(x[:,1]),np.max(x[:,1])])
        plt.show()
        
        pass
    
    # Algoritmoa:
    for t in range(T):
        w[t,:] = 1 / (lamb * (t+1)) * theta[t,:]
        i = random.randint(1,m)
        if y[i-1] * ( np.dot( w[t,:] , x_new[i-1])) < 1:            
            theta[t+1,:] = theta[t,:] + y[i-1]*x_new[i-1] # Konkatenazio hau egiten dugu x'-ren lehen elementua 1 delako
        else:
            theta[t+1,:] = theta[t,:]
    return w,theta,(1/T) * np.sum(w,0),x_new
    

# Adibidea

# Lagina
x = np.array([[2.3,1.2],[-1.7,0.7],[-0.4,-2.3],[-0.4,1.4],[-1.4,-1.2],[0.5,2.6],[0.6,-0.4],[-2.5,1.4],[1.5,-1.5],[-2.6,-0.5]])
y_bek = np.array([1,-1,1,-1,1,-1,1,-1,1,-1,])
# Emaitza
w_hat,theta,w,x_berria = soft_SVM_SGD(x,y_bek,1000,0.5,standardize= True,plot=True)
b = w[0]

x = np.linspace(-3, 3, 100)
y = -b / w[2] - w[1]/w[2]*x
plt.plot(x,y)


plt.scatter(x_berria[:,1],x_berria[:,2],c = y_bek, cmap = "viridis")
plt.grid()
plt.title("SVM-leuna SGD algoritmoarekin")
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.show()


