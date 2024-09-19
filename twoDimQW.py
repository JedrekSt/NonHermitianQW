import numpy as np
from operators import op
from scipy.linalg import expm

class two_dim_QW:
    def __init__(self,dim_x,dim_y,x_coin_dict,y_coin_dict,**kwargs):
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.Evolution(x_coin_dict,y_coin_dict)
        self.state = self.initial_state(**kwargs)

    def Coin(self,**kwargs):
        n = kwargs.get('n')
        th = kwargs.get('th')
        return expm(- 1j * sum(n[i]*op.S[i] for i in range(len(n))) * th / 2)
    
    def Evolution(self,x_coin_dict,y_coin_dict):
        self.Cx = self.Coin(**x_coin_dict)
        self.Cy = self.Coin(**y_coin_dict)

        self.Sx = self.Step(axis = 'x')
        self.Sy = self.Step(axis = 'y')

        Idx = np.eye(self.dim_x)
        Idy = np.eye(self.dim_y)

        self.U_ = self.Sx @ op.n_fold_kron([Idx,Idy,self.Cx]) 
        self.U_ = self.U_ @ self.Sy @ op.n_fold_kron([Idx,Idy,self.Cy]) 
    
    def Step(self,**kwargs):
        if kwargs.get('axis','y')=='x':
            right = op.circ_shift(self.dim_x,k=1)
            left = op.circ_shift(self.dim_x,k=-1)
            Id_ = np.eye(self.dim_y)
            return op.n_fold_kron([right,Id_,op.s_p]) + op.n_fold_kron([left,Id_,op.s_m])
        else:
            right = op.circ_shift(self.dim_y,k=1)
            left = op.circ_shift(self.dim_y,k=-1)
            Id_ = np.eye(self.dim_x) 
            return op.n_fold_kron([Id_,right,op.s_p]) + op.n_fold_kron([Id_,left,op.s_m])

    def initial_state(self,**kwargs):
        state = np.zeros((self.dim_x,))
        state[kwargs.get('x0',self.dim_x//2)] = 1
        state2 = np.zeros((self.dim_y,))
        state2[kwargs.get('y0',self.dim_y//2)] = 1
        state_c = np.array([kwargs.get('a',1.0),kwargs.get('b',1.0)])
        state_c = state_c / np.sqrt(op.dag(state_c) @ state_c)
        return op.n_fold_kron([state,state2,state_c])
    
    def get_prob(self):
        return (np.abs(self.state)**2)[0::2] + (np.abs(self.state)**2)[1::2]

    def evolve(self,steps = 100):
        self.data = [self.get_prob().reshape(self.dim_x,self.dim_y)]
        for _ in range(steps):
            self.state = self.U_ @ self.state
            self.data.append(self.get_prob().reshape(self.dim_x,self.dim_y))
        return np.stack(self.data,axis=0)
    
class diagonal_two_dim_QW(two_dim_QW):
    def Evolution(self,x_coin_dict,y_coin_dict):
        self.Cx = self.Coin(x_coin_dict['th'],x_coin_dict['n'])
        self.Cy = self.Coin(y_coin_dict['th'],y_coin_dict['n'])

        self.Sx = self.Step(axis = 'x')
        self.Sy = self.Step(axis = 'y')

        Idx = np.eye(self.dim_x)
        Idy = np.eye(self.dim_y)
        Id2 = np.eye(2)

        self.U_ = self.Sy @ op.n_fold_kron([Idx,Idy,self.Cy]) 
        self.U_ = self.U_ @ self.Sx @ op.n_fold_kron([Idx,Idy,self.Cx]) 
        self.U_ = self.Sx @ self.Sy @ op.n_fold_kron([Idx,Idy,self.Cx]) @ self.U_  
        