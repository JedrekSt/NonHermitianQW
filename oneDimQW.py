import numpy as np
from operators import op
from numpy.linalg import eigvals
from scipy.linalg import expm

class one_dim_QW:
    def __init__(self,dim,coin_dict,**kwargs):
        self.dim = dim
        self.Evolution(**coin_dict)
        self.state = self.initial_state(**kwargs)

    def Evolution(self,**kwargs):
        n = kwargs.get('n',[0,1,0])
        th = kwargs.get('th',np.pi/2)
        self.C_ = expm(- 1j * sum(n[i]*op.S[i] for i in range(len(n))) * th / 2)
        self.S_ = self.Step()
        self.U_ = self.S_ @ np.kron(np.eye(self.dim),self.C_)
    
    def initial_state(self,**kwargs):
        st = np.zeros(self.dim)
        st[kwargs.get('x0',1)] = 1
        cst = np.array([kwargs.get('a',1),kwargs.get('b',1)])
        cst = cst / np.sqrt( op.dag(cst) @ cst )
        return np.kron(st,cst)
    
    def Step(self):
        right = op.circ_shift(self.dim,k=1)
        left = op.circ_shift(self.dim,k=-1)
        return op.n_fold_kron([right,op.s_p]) + op.n_fold_kron([left,op.s_m])
    
    def get_prob(self):
        prob = np.abs(self.state)**2
        return prob[0::2] + prob[1::2]

    def evolve(self,steps):
        self.data = self.get_prob()
        for _ in range(steps):
            self.state = self.U_ @ self.state
            self.data = np.vstack((self.data,self.get_prob()))
        return self.data
    
    def momentum_U(self):
        T = np.diag(np.exp(-1j*2*np.pi/self.dim * (np.arange(0,self.dim)-self.dim//2)))
        return (np.kron(T,op.s_p)+np.kron(op.dag(T),op.s_m)) @ np.kron(np.eye(self.dim),self.C_) 

class ss_one_dim_QW(one_dim_QW):
    def Evolution(self,**kwargs):
        n = kwargs.get('n',[0,1,0])
        self.th1 = kwargs.get('th1',np.pi/2)
        self.th2 = kwargs.get('th2',np.pi/2)
        self.C_ = lambda th : expm(- 1j * sum(n[i]*op.S[i] for i in range(len(n))) * th / 2)
        self.Sr = np.kron(np.eye(self.dim),op.s_m) + np.kron(op.circ_shift(self.dim,k=1),op.s_p)
        self.Sl = np.kron(np.eye(self.dim),op.s_p) + np.kron(op.circ_shift(self.dim,k=-1),op.s_m)
        self.U_ = self.Sr @ np.kron(np.eye(self.dim),self.C_(self.th1)) @ self.Sl @ np.kron(np.eye(self.dim),self.C_(self.th2))

    def momentum_U(self):
        T = np.diag(np.exp(-1j*2*np.pi/self.dim * np.arange(0,self.dim)))
        right = np.kron(np.eye(self.dim),op.s_m) + np.kron(T,op.s_p)
        left = np.kron(np.eye(self.dim),op.s_p) + np.kron(op.dag(T),op.s_m)
        return right @ np.kron(np.eye(self.dim),self.C_(self.th1)) @ left @ np.kron(np.eye(self.dim),self.C_(self.th2))

    def eigenvals(self):
        E = np.angle(eigvals(self.momentum_U()))
        return E[E>0], E[E<0]

class nonUnitary_QW(ss_one_dim_QW):
    def Evolution(self,**kwargs):
        n = kwargs.get('n',[0,1,0])
        del_ = kwargs.get('del',0.01) 
        self.th1 = kwargs.get('th1',np.pi/2)
        self.th2 = kwargs.get('th2',np.pi/2)
        self.C_ = lambda th : expm(- 1j * sum(n[i]*op.S[i] for i in range(len(n))) * th / 2)
        self.Sr = np.kron(np.eye(self.dim),op.s_m) + np.kron(op.circ_shift(self.dim,k=1),op.s_p)
        self.Sl = np.kron(np.eye(self.dim),op.s_p) + np.kron(op.circ_shift(self.dim,k=-1),op.s_m)
        self.G_ = np.kron(np.eye(self.dim),expm(del_ * op.sz))
        self.G_p = np.kron(np.eye(self.dim),expm(-del_ * op.sz))
        self.U_ = self.Sr @ self.G_ @ np.kron(np.eye(self.dim),self.C_(self.th1)) @ self.Sl @ self.G_p @ np.kron(np.eye(self.dim),self.C_(self.th2))

    def momentum_U(self):
        T = np.diag(np.exp(-1j*2*np.pi/self.dim * np.arange(0,self.dim)))
        right = np.kron(np.eye(self.dim),op.s_m) + np.kron(T,op.s_p)
        left = np.kron(np.eye(self.dim),op.s_p) + np.kron(op.dag(T),op.s_m)
        return right @ self.G_ @ np.kron(np.eye(self.dim),self.C_(self.th1)) @ left @ self.G_p @ np.kron(np.eye(self.dim),self.C_(self.th2))
