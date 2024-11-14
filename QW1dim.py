import numpy as np
from operators import op
from oneDimQW import one_dim_QW as odQW
from numpy.linalg import eigvals
from scipy.linalg import expm
from abc import ABC, abstractmethod

class TopologicalPhaseQW(odQW):
    def Evolution(self,**kwargs):
        thA = kwargs.get('thA',np.pi/4)
        thB = kwargs.get('thB',np.pi/4)
        self.C_ = np.kron(np.diag([0 if i > self.dim//2 else 1 for i in range(self.dim)]),expm(- 1j * op.sy * thA / 2))
        self.C_ += np.kron(np.diag([0 if i <= self.dim//2 else 1 for i in range(self.dim)]),expm(- 1j * op.sy * thB / 2))
        self.S_ = self.Step()
        return self.S_ @ self.C_

class BoundaryQuantumWalk(odQW):
    def Evolution(self,**kwargs):
        th = kwargs.get('th',np.pi/4)
        th_bound = np.pi/2 * (1 if th < 0 else -1) 
        self.C_ = np.kron(np.diag([0 if i == 0 or i== self.dim-1 else 1 for i in range(self.dim)]),expm(- 1j * op.sy * th ))
        self.C_ += np.kron(np.diag([1 if i == 0 or i== self.dim-1 else 0 for i in range(self.dim)]),expm(- 1j * op.sy * th_bound ))
        self.S_ = self.Step()
        return self.S_ @ self.C_
    
class PhaseDefect(odQW):
    def Evolution(self,**kwargs):
        th = kwargs.get('th',np.pi/4)
        th_bound = np.pi/2 * (1 if th < 0 else -1) 
        self.C_ = np.kron(np.diag([0 if i == self.dim//2 else 1 for i in range(self.dim)]),expm(- 1j * op.sy * th ))
        self.C_ += np.kron(np.diag([1 if i == self.dim//2 else 0 for i in range(self.dim)]),expm(- 1j * op.sy * th_bound ))
        self.S_ = self.Step()
        return self.S_ @ self.C_
    
class PhaseDefect_random(odQW):
    def Evolution(self,**kwargs):
        th = kwargs.get('th',np.pi/4)
        th_bound = np.pi/2 * (1 if th < 0 else -1) 
        defect = np.random.randint(self.dim)
        self.C_ = np.kron(np.diag([0 if i == defect else 1 for i in range(self.dim)]),expm(- 1j * op.sy * th ))
        self.C_ += np.kron(np.diag([1 if i == defect else 0 for i in range(self.dim)]),expm(- 1j * op.sy * th_bound ))
        self.S_ = self.Step()
        return self.S_ @ self.C_
    
    def evolve(self,steps,**kwargs):
        self.data = self.get_prob()
        for _ in range(steps):
            self.U_ = self.Evolution(**kwargs)
            self.state = self.U_ @ self.state
            self.data = np.vstack((self.data,self.get_prob()))
        return self.data
    

class oneDqw:
    def __init__(self,dim,coin_dict,**kwargs):
        self.dim = dim
        self.U_ = self.Evolution(**coin_dict)
        self.state = self.initial_state(**kwargs)
    
    def Evolution(self,**kwargs):
        self.S_r,self.S_l = self.Step()
        self.C_r,self.C_l = self.Coin(**kwargs)
        return self.S_l @ self.C_l @ self.S_r @ self.C_r

    def Step(self):
        right = op.circ_shift(self.dim,k=-1)
        left = op.circ_shift(self.dim,k=1)
        return np.kron(right,op.s_p) + np.kron(np.eye(self.dim),op.s_m), np.kron(left,op.s_m) + np.kron(np.eye(self.dim),op.s_p)

    @abstractmethod
    def Coin(self,**kwargs):
        pass

    def initial_state(self,**kwargs):
        st = np.zeros(self.dim)
        st[kwargs.get('x0',self.dim//2)] = 1
        cst = np.array([kwargs.get('a',1),kwargs.get('b',1j)])
        cst = cst / np.sqrt( op.dag(cst) @ cst )
        return np.kron(st,cst)
    
    def Eigen(self):
        E,U = np.linalg.eig(self.U_)
        inds = np.argsort(np.angle(E))
        return np.angle(E)[inds],U[:,inds]

class boundaryQW(oneDqw):
    def Coin(self,**kwargs):
        right = expm(- 1j * op.sy * kwargs.get("th_r",np.pi/4))
        left = expm(- 1j * op.sy * kwargs.get("th_l",np.pi/4))
        proj_ = lambda k: np.kron(np.eye(self.dim,1,-k),np.eye(self.dim,1,-k).flatten())
        right_coin = np.kron(np.eye(self.dim) - proj_(self.dim-1) - proj_(0) ,right) + np.kron(proj_(self.dim-1),op.sx) + np.kron(proj_(0),op.Id)
        left_coin = np.kron(np.eye(self.dim) - proj_(self.dim-1) - proj_(0) ,left) + np.kron(proj_(0),op.sx) + np.kron(proj_(self.dim-1),op.Id)
        return left_coin,right_coin