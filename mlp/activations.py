import numpy as np 

class activations:
    def __init__(self):
        pass

    def derivada_relu(self, z):
        return (z > 0).astype(float) | # soh a derivada da relu mesmo

    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        e = np.exp(z) 
        return e/np.sum(e)
    

