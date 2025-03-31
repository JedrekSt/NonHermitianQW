import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from scipy.linalg import expm
from numpy.linalg import eig
from IPython.display import clear_output
from qw_modules.operators import op


class synch_QW:
    def __init__(self,dim = 10,steps=200):
        self.dim = dim
        self.steps = steps

    def Coin(self,y,**kwargs):
        assert 'th' in kwargs.keys(), "th for coin must be specified"
        assert 'n' in kwargs.keys(), "n for coin must be specified"
        assert (isinstance(kwargs['n'],list) or isinstance(kwargs['n'],np.ndarray)) and len(kwargs['n'])==3, "n has invalid shape or dtype"

        C_ = expm(-1j * sum(op.S[i] * kwargs['n'][i] for i in range(len(op.S)))*kwargs['th']) 
        proj = lambda x_0,dim_ : np.diag(np.eye(1,dim_,k=x_0 % dim_).flatten()) 
        return sum(np.kron(np.kron(np.eye(self.dim) - proj(y[i],self.dim) - proj(y[i] + 1,self.dim),C_) + np.kron(proj(y[i],self.dim)+proj(y[i]+1,self.dim),op.sx),proj(i,len(y))) for i in range(len(y)))
    
    def Step(self):
        return np.kron(op.circ_shift(self.dim,-1),(op.Id + op.sz)/2) + np.kron(op.circ_shift(self.dim,1),(op.Id - op.sz)/2) 

    def Krauss(self,y,p):
        K0 = np.sqrt(p)*np.kron(np.diag([1 if i in y else 0 for i in range(self.dim)]),np.kron(op.sp,np.eye(len(y))))
        K1_ = np.eye(self.dim*2*len(y)) - op.dag(K0) @ K0

        En, Uv = eig(K1_)
        K1 = Uv @ np.diag(np.sqrt(En)) @ op.dag(Uv)
        return K0,K1
    
    def psi0_(self,x0,a,b,y_):
        p0 = np.zeros(self.dim)
        p0[x0] = 1
        p0 = p0/ np.sqrt(p0 @ p0)
        return np.kron(np.kron(p0,np.array([a,b])/np.sqrt(abs(a)**2+abs(b)**2)),y_)

    def prob(self,rho_):
        dims = self.y_dim*2
        return np.diag(sum(rho_[i::dims,i::dims] for i in range(dims)))

    def evolve(self,y,y_,p,x0,coin_kwargs):
        self.y_dim = len(y)
        self.rho_eigen = []
        a=1
        b=0
        psi0 = self.psi0_(x0,a,b,y_)
        rho_ = np.kron(psi0.reshape(-1,1),np.conj(psi0))

        self.C_ = self.Coin(y,**coin_kwargs)
        self.S_ = np.kron(self.Step(),np.eye(len(y)))
        self.K0_, self.K1_ = self.Krauss(y,p)

        data = self.prob(rho_)
        negativity_ = [self.negativity(self.partial_transpose(rho_))]
        prob = [np.diag(self.rabbit_DM(rho_))]
        for i in range(self.steps):
            
            print("*" * ((i*30)//(self.steps-1)) + "-" * (30 - (i*30)//(self.steps-1)))
            print("DM trace: ",np.real(np.trace(rho_)))
            print(f"{np.round(i/(self.steps-1),4)*100}% completed")

            rho_ = self.S_ @ self.C_ @ (self.K0_ @ rho_ @ op.dag(self.K0_) +self.K1_ @ rho_ @ op.dag(self.K1_)) @ op.dag(self.C_) @ op.dag(self.S_)
            data = np.vstack((data,self.prob(rho_)))
            negativity_.append(self.negativity(self.partial_transpose(rho_)))
            prob.append(np.diag(self.rabbit_DM(rho_)))
            clear_output(wait=True)
        return data,np.array(negativity_),np.array(prob)

    def partial_transpose(self,rho_):
        dims = self.dim // self.y_dim
        hsplit_ = [[np.transpose(el2_) for el2_ in np.hsplit(el_,dims)] for el_ in np.vsplit(rho_,dims)]
        return np.vstack(tuple(np.hstack(tuple(el_ for el_ in el2_)) for el2_ in hsplit_))
    
    def negativity(self,part_rho):
        E = np.real(eig(part_rho)[0])
        self.rho_eigen.append(E)
        return len(E[E< -1e-2])
    
    def rabbit_DM(self,rho_):  
        dim = rho_.shape[0]
        dim2 = self.y_dim
        dim1 = dim // self.y_dim
        A_ = rho_.reshape(dim1,dim2,dim1,dim2)
        return np.trace(A_,axis1=0,axis2=2)
    
class circ:
    def __init__(self,dim):
        self.edges = [(i ,(i+1)%dim) for i in range(dim)]
        self.graph = nx.Graph()
        self.vert = self.graph.nodes
        self.graph.add_edges_from(self.edges)
        self.dim = dim

    def show_g(self,y,y_state):
        fig,axs = plt.subplots(1,2,figsize=(13,6))
        pos = nx.circular_layout(self.graph)

        axs[0].set_title("cicle")
        colors = list(map(lambda el : 'red' if (el in y ) else 'lightblue',self.vert))
        ecolors = list(map(lambda el : 'blue' if (el in y or el-1 in y) else 'white',self.vert))
        nx.draw(
            self.graph,
            pos,with_labels = True,
            node_size = 400,
            font_size = 11,
            node_color=colors,
            edgecolors = ecolors,
            ax=axs[0]
        )
        for lab in y:
            axs[0].text(*self.text_position_(lab,pos), 'rabbit', fontsize=11, color='red', ha='center')

        axs[1].set_title(f"initial rabbit superposition {y_state}")
        axs[1].bar(y,np.abs(y_state)**2,color="orange")
        axs[1].set_xlim(0,self.dim)
        axs[1].set_xlabel("position")
        axs[1].set_ylabel("probability")

        plt.show()

    def text_position_(self,y_,pos):
        text_pos = (y_ // (self.dim // 4))
        shift_x = 0.05
        shift_y = 0.15
        if text_pos == 0:
            return (pos[y_][0]-shift_x,pos[y_][1]-shift_y)
        elif text_pos == 1:
            return (pos[y_][0]+shift_x,pos[y_][1]-shift_y)
        elif text_pos == 2:
            return (pos[y_][0]+shift_x,pos[y_][1]+shift_y)
        elif text_pos == 3:
            return (pos[y_][0]-shift_x,pos[y_][1]+shift_y)

