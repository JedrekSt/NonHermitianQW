import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("NonHermitianQW"))))

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import eig
from scipy.linalg import expm
from operators import op
from abc import ABC, abstractmethod

class SSH(ABC):

    def __init__(self,dim,chain_dict):
        self.dim = dim
        self.H_ = self.Hamiltonian(**chain_dict)

    @abstractmethod
    def Hamiltonian(self):
        pass

    def get_eigen(self):
        return eig(self.H_)[0]
    
    def Evolution(self,psi0,T,dt = 0.01):
        t_ = np.arange(dt,T,dt)
        data = np.abs(psi0)**2
        psi = psi0.copy()
        Udt = expm(-1j*self.H_*dt)
        for i in range(len(t_)):
            psi = Udt @ psi 
            data = np.vstack((data,np.abs(psi)**2))
        return data
    
    def spectrum_dependencies(self):
        u_ = np.arange(0,1.05,0.05)
        v_ = np.arange(0,1.05,0.05)
        return u_,v_,np.stack(
            [np.stack([np.sort(eig(self.Hamiltonian(**{"u":u, "v":v}))[0]) for v in v_],axis=1) for u in u_],
            axis=2)

class BulkSSH(SSH):

    def Hamiltonian(self,**kwargs):
        self.u = kwargs.get("u")
        self.v = kwargs.get("v")
        left = np.kron(np.eye(self.dim),op.sp) + np.kron(np.eye(self.dim),op.sm)
        right = np.kron(op.circ_shift(self.dim,k=1),op.sp) + np.kron(op.circ_shift(self.dim,k=-1),op.sm)
        return self.u * left + self.v * right
    
    def get_momentum(self):
        QFT = np.kron(op.QFT(self.dim),np.eye(2))
        Hk = op.dag(QFT) @ self.H_ @ QFT
        return Hk
    
    def get_eigen_momentum(self):
        E = []
        Hk = self.get_momentum()
        for i in range(self.dim):
            E.append(eig(Hk[2*i : 2*i+2, 2*i : 2*i+2])[0])
        return np.array(E).T

class BoundSSH(SSH):

    def Hamiltonian(self,**kwargs):
        self.u = kwargs.get("u")
        self.v = kwargs.get("v")
        left = np.kron(np.eye(self.dim),op.sp) + np.kron(np.eye(self.dim),op.sm)
        right = np.kron(np.eye(self.dim,k=1),op.sp) + np.kron(np.eye(self.dim,k=-1),op.sm)
        return self.u * left + self.v * right