import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm
from matplotlib import pyplot as plt
from qw_modules.operators import op

class rabbit_wout_def:
    def __init__(self,dim,coin_dict,rabbit_dict,init_state_dict):
        self.dim = dim
        self.C_ = self.Coin(**coin_dict)
        self.S_ = self.Step()
        self.V_ = self.Defect(**rabbit_dict)
        
        self.U_ = self.V_ @ self.S_ @ self.C_
        self.psi_ = self.state(**init_state_dict)

    def state(self,**kwargs):
        x0 = kwargs.get("x0",self.dim//2)
        a = kwargs.get("a",1)
        b = kwargs.get("b",1j)
        st = np.kron(np.eye(M=self.dim,N=1,k=x0).flatten(),np.array([a,b]))
        return st / np.sqrt(np.conj(st) @ st)

    def Coin(self,**kwargs):
        return np.kron(np.eye(self.dim),expm(-1j * op.sy * kwargs.get("theta",np.pi/4)))
    
    def Step(self):
        right = op.circ_shift(self.dim,k=1)
        left = op.circ_shift(self.dim,k=-1)
        return op.n_fold_kron([right,op.s_p]) + op.n_fold_kron([left,op.s_m])
    
    def Defect(self,**kwargs):
        y_ = kwargs.get("y_pos",[self.dim//3])
        ps = kwargs.get("ps",np.pi/5)
        return np.kron(np.diag(np.exp(-1j * np.array([ 0 if j in y_ else ps for j in range(self.dim)]))),np.eye(2))
    
    def get_eigen(self):
        E,U = eig(self.U_)
        return np.angle(E),U
    
    def get_prob(self):
        return (np.abs(self.psi_)**2)[0::2]+(np.abs(self.psi_)**2)[1::2]
    
    def Evolve(self,steps):
        data_ = self.get_prob()
        for _ in range(steps):
            self.psi_ = self.U_ @ self.psi_
            data_ = np.vstack((data_ , self.get_prob()))
        return data_
    
    def Get_evolution_with_plot(self,steps,width,length):
        data = self.Evolve(steps)
        
        fig = plt.figure(figsize = (width,length))
        ax = fig.add_subplot()

        ax.imshow(data,origin="lower",cmap = "viridis",aspect = "auto")
        ax.set_xlabel("position")
        ax.set_ylabel("time")
        ax.set_title("QW qith defect")
        return fig,ax
    
    def Get_eigen_vectors_with_plot(self):
        cols = 5
        rows = self.dim // 4 + 1
        scale = 5
        fig = plt.figure(figsize = (cols * scale,rows * scale))

        En,U_ = self.get_eigen()
        for i in range(len(En)//2):
            eig1 = np.abs(U_[:,2*i])**2
            eig2 = np.abs(U_[:,2*i+1])**2
            ax = fig.add_subplot(rows,cols,i+1)
            ax.plot(range(self.dim),eig1[0::2]+eig1[1::2],color = "red",alpha = 0.7,lw = 2)
            ax.plot(range(self.dim),eig2[0::2]+eig2[1::2],color = "blue",alpha = 0.7,lw = 2)
        return fig,ax


class rabbit_trap(rabbit_wout_def):
    def __init__(self,dim,coin_dict,rabbit_dict,init_state_dict):
        super().__init__(dim,coin_dict,rabbit_dict,init_state_dict)
        self.K0,self.K1 = self.Krauss(**rabbit_dict)
        self.rho_ = np.kron(self.psi_.reshape(-1,1),np.conj(self.psi_))
    
    def get_prob(self):
        return np.abs(np.diag(self.rho_)[0::2] + np.diag(self.rho_)[1::2])
    
    def Krauss(self,**kwargs):
        y_ = kwargs.get("y_pos",[self.dim//3])
        prob = kwargs.get("p",1)
        K0 = np.sqrt(prob) * sum(np.kron(np.diag([0 if i != j else 1 for i in range(self.dim)]),[op.s_p,op.s_m][j % 2]) for j in range(len(y_)))
        K1_ = np.eye(self.dim*2) - op.dag(K0) @ K0
        En, Uv = eig(K1_)
        K1 = Uv @ np.diag(np.sqrt(En)) @ op.dag(Uv)
        return K0,K1
    
    def Evolve(self,steps):
        data_ = self.get_prob()
        for _ in range(steps):
            temp = (self.U_ @ self.rho_ @ op.dag(self.U_))
            self.rho_ = self.K1 @ temp @ op.dag(self.K1) + self.K0 @ temp @ op.dag(self.K0)
            data_ = np.vstack((data_ , self.get_prob()))
        return data_






    

    



    