import numpy as np

class op:
    sx = np.array([
        [0,1],
        [1,0]
    ])
    sy = np.array([
        [0,-1j],
        [1j,0]
    ])
    sz = np.array([
        [1,0],
        [0,-1]
    ])

    Id = np.eye(2)

    sp = np.array([
        [0,0],
        [1,0]
    ])
    sm = np.array([
        [0,1],
        [0,0]
    ])

    s_p = (np.eye(2) + sz)/2 
    s_m = (np.eye(2) - sz)/2 
    
    S = (sx,sy,sz)

    @staticmethod
    def circ_shift(dim,k):
        return np.eye(dim,k=k) + np.eye(dim,k = ((-dim + k)  if k>=0 else (dim+k)))
    
    @staticmethod
    def whole_step(dim):
        return op.circ_shift(dim,k=1) + op.circ_shift(dim,k=-1)
    
    @staticmethod
    def dag(A):
        return np.conj(A.T)
    
    @staticmethod
    def n_fold_kron(op_tab):
        ans = op_tab[0]
        for i in range(1,len(op_tab)):
            ans = np.kron(ans,op_tab[i])
        return ans
    
    @staticmethod
    def QFT(dim):
        om = np.exp(-1j*2*np.pi/dim)
        single_vec = np.array([om ** n for n in range(dim)]).reshape(-1,1)
        return np.hstack(tuple(single_vec**m for m in range(dim)))/np.sqrt(dim)
    
    @staticmethod
    def proj(dim,x):
        return np.diag(np.array([0 if i != x else 1 for i in range(dim)]))
    