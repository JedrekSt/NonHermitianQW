import numpy as np
from operators import op
from scipy.linalg import expm
from numpy import pi

class single_coin_two_dim_QW:
    def __init__(self,dim_x,dim_y,x_coin_dict,y_coin_dict,**kwargs):
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.U_ = self.Evolution(x_coin_dict,y_coin_dict)
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

        U_ = self.Sx @ op.n_fold_kron([Idx,Idy,self.Cx]) 
        U_ = U_ @ self.Sy @ op.n_fold_kron([Idx,Idy,self.Cy]) 
        return U_
    
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
    
class diagonal_two_dim_QW(single_coin_two_dim_QW):
    def Evolution(self,x_coin_dict,y_coin_dict):
        self.Cx = self.Coin(x_coin_dict['th'],x_coin_dict['n'])
        self.Cy = self.Coin(y_coin_dict['th'],y_coin_dict['n'])

        self.Sx = self.Step(axis = 'x')
        self.Sy = self.Step(axis = 'y')

        Idx = np.eye(self.dim_x)
        Idy = np.eye(self.dim_y)
        Id2 = np.eye(2)

        U_ = self.Sy @ op.n_fold_kron([Idx,Idy,self.Cy]) 
        U_ = U_ @ self.Sx @ op.n_fold_kron([Idx,Idy,self.Cx]) 
        U_ = self.Sx @ self.Sy @ op.n_fold_kron([Idx,Idy,self.Cx]) @ U_  

        return U_
        

class double_coin_two_dim_QW:
    def __init__(self,x_dim,y_dim,x_coin_dict,y_coin_dict,**kwargs):
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.Idx = np.eye(self.x_dim)
        self.Idy = np.eye(self.y_dim)
        self.Id2 = np.eye(2)

        self.U_ = self.Evolution(x_coin_dict,y_coin_dict)
        self.U_k = self.momentum_U()
        self.psi = self.initial_state(**kwargs)

    def Evolution(self,x_coin_dict,y_coin_dict):
        self.Cx = self.Coin(**x_coin_dict)
        self.Cy = self.Coin(**y_coin_dict)

        self.S_ = self.Step()

        return self.S_ @ op.n_fold_kron([self.Idx,self.Idy,self.Cx,np.eye(2)]) @ op.n_fold_kron([self.Idx,self.Idy,np.eye(2),self.Cy])
    
    def Coin(self,**kwargs):
        n_ = kwargs.get('n',[0,1,0])
        th = kwargs.get('th',pi/8)
        return expm(-1j*sum(n_[i] * op.S[i] for i in range(len(n_)))*th/2)
    
    def Step(self):
        Sx = op.n_fold_kron([op.circ_shift(self.x_dim,k=1),self.Idy,op.s_p,self.Id2])
        Sx += op.n_fold_kron([op.circ_shift(self.x_dim,k=-1),self.Idy,op.s_m,self.Id2])

        Sy = op.n_fold_kron([self.Idx,op.circ_shift(self.y_dim,k=1),self.Id2,op.s_p])
        Sy += op.n_fold_kron([self.Idx,op.circ_shift(self.y_dim,k=-1),self.Id2,op.s_m])

        return Sx @ Sy
    
    def initial_state(self,**kwargs):
        x0 = kwargs.get("x0",self.x_dim//2)
        y0 = kwargs.get("y0",self.y_dim//2)
        ax = kwargs.get("ax",1)
        bx = kwargs.get("bx",1j)
        ay = kwargs.get("ay",1)
        by = kwargs.get("by",1j)
        psix = np.eye(1,self.x_dim,k=x0)[0]
        psiy = np.eye(1,self.y_dim,k=y0)[0]
        c_x = np.array([ax,bx])
        c_y = np.array([ay,by])

        psi = op.n_fold_kron([psix,psiy,c_x,c_y])
        return psi / np.sqrt(op.dag(psi) @ psi)
    
    def get_Prob(self):
        return (np.abs(sum(self.psi[i::4] for i in range(4)))**2).reshape(self.x_dim,self.y_dim)
    
    def Evolve(self,steps):
        data = [self.get_Prob()]
        for _ in range(steps):
            self.psi = self.U_ @ self.psi
            data.append(self.get_Prob())
        return np.stack(data,axis=0)
    
    def momentum_U(self):
        Tx = np.diag(np.exp(-1j* 2*pi/self.x_dim * np.arange(0,self.x_dim)))
        Ty = np.diag(np.exp(-1j* 2*pi/self.y_dim * np.arange(0,self.y_dim)))

        Step_x = op.n_fold_kron([Tx,self.Idy,op.s_p,np.eye(2)]) + op.n_fold_kron([op.dag(Tx),self.Idy,op.s_m,np.eye(2)])
        Step_y = op.n_fold_kron([self.Idx,Ty,np.eye(2),op.s_p]) + op.n_fold_kron([self.Idx,op.dag(Ty),np.eye(2),op.s_m])

        return Step_x @ Step_y @ op.n_fold_kron([self.Idx,self.Idy,self.Cx,np.eye(2)]) @ op.n_fold_kron([self.Idx,self.Idy,np.eye(2),self.Cy])




    

    

