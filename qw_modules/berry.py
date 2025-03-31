import math
import numpy as np
from numpy.linalg import eig
from qw_modules.operators import op

class BerryPhase:
    def __init__(self):
        pass
        
    def Calculate(self,model):
        dim = model.dim
        E,Ut = eig(model.momentum_U())
        E = np.angle(E)
        Ut = np.hstack(tuple(Ut[2*i : 2*i+2,2*i : 2*i+2] for i in range(dim)))
        Mat_m = (op.dag(Ut[:,E<=0]) @ Ut[:,E<=0])
        Mat_p = (op.dag(Ut[:,E>0]) @ Ut[:,E>0])
        ans_m =  np.abs(np.angle(math.prod(np.diag(Mat_m,k=1))*Mat_m[-1,0]))
        ans_p =  np.abs(np.angle(math.prod(np.diag(Mat_p,k=1))*Mat_p[-1,0]))
        return ans_m,ans_p

    def vector_representation(self,model):
        dim = model.dim
        _,Ut = eig(model.momentum_U())
        Ut = np.hstack(tuple(Ut[2*i : 2*i+2,2*i : 2*i+2] for i in range(dim)))
        Ut = Ut / np.sqrt(np.diag(op.dag(Ut)@Ut))
        ans = lambda ax: np.vstack(tuple( np.diag(op.dag(Ut[:,ax::2]) @ S_ @ Ut[:,ax::2]) for S_ in op.S))
        return ans(0),ans(1)
    
