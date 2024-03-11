import random
import numpy as np
import matplotlib.pyplot as plt

def polynomial_kernel(xi,xj,deg = 2):
    return (np.dot(xi,xj) + 1)**2

def gaussian_kernel(xi,xj):
    pass


def soft_SVM_SGD_Kernel(x,y,kernel = "poly",deg = 2,T = 1000,lamb = 1,standardize = True, plot = False,lim = 5):
    """
    x = m luzeerako bektore multzoa (domeinua)
    y = m luzeerako bektorea (izenak)
    lamb = lambda parametroa
    T = iterazio kopurua
    """
    if kernel == "poly":
        K = polynomial_kernel
    elif kernel == "gaussian_kernel":
        K = gaussian_kernel

    # Beharrezko aldagaiak sortu:
    m = len(x)
    d = len(x[0]) + 1 # d + 1 dimentsioan lan egingo dugu, hiperplano homogeneoa bilatuz: w' = (b,w1,w2...,wd) eta x' = (1,x1,x2,...,xd)
    alpha = np.zeros([T,m])
    beta = np.zeros([T+1,m])
    x_new =  np.concatenate( (np.ones((m,1)),x) , axis = 1)
    
    # Algoritmoa:
    for t in range(T):
        alpha[t,:] = 1 / (lamb * (t+1)) * beta[t,:]
        i = random.randint(1,m)

        print("----")
        beta_i = beta[t,i-1]
        print(beta_i)
        beta[t+1,:] = beta[t,:]
        print(beta_i)
        print("----")
        
        kernels = np.zeros(m)
        for j in range(m):
            kernels[j] = K(x[i-1],x[j]) # Hau egiteko modu efizienteago bat?
        print(kernels)
        if y[i-1]*np.dot(alpha[t,:],kernels) < 1:            
            beta[t+1,i-1] = beta_i + y[i-1] 
        else:
            beta[t+1,i-1] = beta[t,i-1]

    return (1/T) * np.sum(alpha,0)
    



# Adibidea

# Lagina
x = np.array([[2.3,1.2],[-1.7,0.7],[-0.4,-2.3],[-0.4,1.4],[-1.4,-1.2],[0.5,2.6],[0.6,-0.4],[-2.5,1.4],[1.5,-1.5],[-2.6,-0.5]])
y_bek = np.array([1,-1,1,-1,1,-1,1,-1,1,-1,])
# Emaitza

print( np.dot( [ 1.5, -1.5],[2.3, 1.2] ))

alpha_txap = soft_SVM_SGD_Kernel(x,y_bek)
print(alpha_txap)


