import numpy as np
from operators import op
from numpy.linalg import eigvals
from scipy.linalg import expm
from oneDimQW import one_dim_QW
import matplotlib.pyplot as plt

class EMqw(one_dim_QW):

    def gen_EM(self,E,B):
        return np.diag([np.exp(1j* complex((E+(-1 if i%2 != 0 else 1) * B ) * 2*np.pi/self.dim * (i//2))) for i in range(self.dim*2)])


    def get_data_EM(self):
        E = np.arange(0,self.dim,dtype = complex)
        B = np.arange(0,self.dim,dtype = complex)
        res =np.zeros((len(E),len(B)),dtype=complex)
        
        for e_ in range(len(E)):
            for b_ in range(len(B)):
                res[e_,b_] = np.trace(op.dag(self.U_) @ self.gen_EM(E[e_],B[b_]))
        return res
    
if __name__ == "__main__":
    dim = 10
    coin_dict = {}
    model = EMqw(dim,coin_dict)
    data = model.get_data_EM()
    
    plt.imshow(np.abs(data))
    print(np.abs(data))
    plt.show()

                
    