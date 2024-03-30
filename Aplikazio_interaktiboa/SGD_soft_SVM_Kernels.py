import random
import numpy as np
import math

def polynomial_kernel(xi,xj,deg = 2):
    return (np.dot(xi,xj) + 1)**deg

def gaussian_kernel(xi,xj,sigma = 0.1):
    return math.exp( (-1) * (np.dot(xi-xj,xi-xj))/ (2*sigma) )

def linear_kernel(xi,xj, k = 0):
    return np.dot(xi,xj) + k

class Nire_SGD_kernelekin:
    def __init__(self,koeficient,kernel):
        self.entrenatuta = False
        self.klaseanitz = False
        self.alpha = None # Modeloaren entrenatu ondorengo aldagaiak hemen egongo dira
        self.X = None
        self.Y_bakarrak = None
        self.m = None
        self.koeficient = 1 / koeficient # Scikit-Learn paketearen baliokidea izateko
        self.nota = 0
        
        # Kernel aukeraketa:
        if kernel == "kernel gaussiarra":
            self.kernel = gaussian_kernel
        elif kernel == "kernel polinomiala":
            self.kernel = polynomial_kernel
        elif kernel == "kernel lineala":
            self.kernel = linear_kernel
        else:
            print("Sartu duzun kernela ez da zuzena!")
            self.kernel = gaussian_kernel

    def kernelak_kalkulatu(self,x):
        return [self.kernel(xi,x) for xi in self.X]
        
    def fit(self,X,Y,iter = 10000):
        """
        Algoritmoa entrenatzen du X bektore eta Y izeneko lagin batekin.
        """
        self.m = len(X)
        self.X =  np.concatenate( (np.ones((self.m,1)),X) , axis = 1) # Ondoren predikzioak egiteko gorde egingo dugu klasearen barruan
        if len(np.unique(Y)) == 2: # Klasifikazio dikotomikoa
            T = iter
            alpha = np.zeros([T,self.m])
            beta = np.zeros([T+1,self.m]) # T+1 jartzen dugu azken iterazioan arazorik ez egoteko, hala ere ez dugu iterazio horko informazioa erabiliko
            # Algoritmoa:
            for t in range(T):
                # 1. Pausua:
                alpha[t,:] = 1 / (self.koeficient * (t+1)) * beta[t,:] # t+1 egiten dugu Pythonen indizeak 0-n hasten direlako
                # 2. Pausua:
                i = random.randint(1,self.m) - 1 # -1 egiten dugu indizeen arazoa konpontzeko
                # 3. Pausua:
                beta[t+1,:] = beta[t,:] # Oraingoz beta_i^(t+1) = beta_i^(t) da, gero aldatuko dugu 
                # 4. Pausua, kernelak kalkulatu: 
                kernels = self.kernelak_kalkulatu(self.X[i])
                if (Y[i]*np.dot(alpha[t,:],kernels) < 1):            
                    beta[t+1,i] = beta[t,i] + Y[i] 
                else:
                    beta[t+1,i] = beta[t,i]
            self.alpha =  (1/T) * np.sum(alpha,0) # Predikzioak egiteko definitu

        else: # Klasifikazio anitzkoitza
            self.klaseanitz = True
            self.alpha = [] # k klaserako, k hipotesi desberdinen 'outputak'
            self.Y_bakarrak = np.unique(Y)
            Y_desberdinak = []
            for izena in self.Y_bakarrak:
                Y_desberdinak.append([1 if elementua == izena else -1 for elementua in Y])
            
            r = 0
            for Y_desb in Y_desberdinak:
                # Entrenatu (goiko kode berdina)
                T = iter
                alpha = np.zeros([T,self.m])
                beta = np.zeros([T+1,self.m])
                for t in range(T):
                    alpha[t,:] = 1 / (self.koeficient * (t+1)) * beta[t,:]
                    i = random.randint(1,self.m) - 1
                    beta[t+1,:] = beta[t,:] 
                    kernels = self.kernelak_kalkulatu(self.X[i])
                    if (Y_desb[i]*np.dot(alpha[t,:],kernels) < 1):            
                        beta[t+1,i] = beta[t,i] + Y_desb[i] 
                    else:
                        beta[t+1,i] = beta[t,i]
                self.alpha.append((1/T) * np.sum(alpha,0))
                r+=1
                print(f"Fitted class number {r} of {len(self.Y_bakarrak)}")
            
        self.entrenatuta = True
        print("Fitted!")

    def predict(self,x):
        """
        x domeinuko elementu berri batentzako, jada entrenatutako modeloak ematen duen x-ren izena itzuliko du.
        """
        x_predict = np.concatenate(([1],x))
        if self.entrenatuta and not self.klaseanitz: # Klasifikazio normala (Y = {+1, -1})
            kernelak = self.kernelak_kalkulatu(x_predict)
            balioa = np.dot(self.alpha,kernelak)
            return 1 if balioa > 0 else -1
        elif self.entrenatuta and self.klaseanitz: # Klasifikazio anitzkoitza
            kernelak = self.kernelak_kalkulatu(x_predict)
            balioak = []
            for alpha in self.alpha:
                balioak.append(np.dot(alpha,kernelak))
            return self.Y_bakarrak[np.argmax(balioak)]
        else:
            print("Modeloa oraindik ez da entrenatu")
    
    def predict_anitzkoitza(self,x_multzoa):
        labels = np.zeros(len(x_multzoa))
        for i,x in enumerate(x_multzoa):
            labels[i] = self.predict(x)
        return labels

    def score(self,X,Y):
        """
        X bektoreko eta Y izeneko datu-multzo bat emanik, modeloak dauden datu guztietatik ongi klasifikatzen dituen proportzioa itzuliko du
        """
        if self.entrenatuta:
            m = len(X)
            klasifikazio_zuzen_kopurua = 0
            for i,bektore in enumerate(X):
                if self.predict(bektore) == Y[i]:
                    klasifikazio_zuzen_kopurua += 1
            self.nota = klasifikazio_zuzen_kopurua / m
            return self.nota
        print("Modeloa oraindik ez da entrenatu")

