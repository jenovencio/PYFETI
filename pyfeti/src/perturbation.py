import numpy as np
from scipy.sparse import linalg
from scipy import sparse
from .linalg import LinearSys, create_voigt_rotation_matrix

def perturbation(shape,sigma=1.0,mu=0.0,seed=None):
    np.random.seed(seed=seed)
    return sigma*np.random.randn(*shape) + mu

def eigvec_perturbation(V,seed=1,sigma=1, mu = 0, symmetric=True,diagonal=False,sparseout=True):
    n, k = V.shape  
    if diagonal:
        deltaA = np.diag(self.perturbation((k,),seed=seed))
    else:
        deltaA = perturbation((k,k),sigma=sigma, mu = mu ,seed=seed)

    if symmetric:
        deltaA = deltaA.T.dot(deltaA)
    if sparseout:
        return sparse.csc_matrix(V.dot(deltaA.dot(V.T)))
    else:
        return V.dot(deltaA.dot(V.T))
    
    
class Perturbation():
    def __init__(self,A,sigma=1,mu=0):
        self.A = A
        self.sigma = sigma
        self.mu = mu
        self.deltaA = None
        self.k = self.A.shape[0]
        
    def eigvec_perturbation(self,k=None,seed=1,symmetric=True,diagonal=False):
        if k is None:
            k = self.k
            
        try:
            self.eigval, self.V = linalg.eigs(self.A,k=k)
        except:
            self.eigval, self.V = np.linalg.eig(self.A)
        
        new_id = np.argsort(self.eigval)[::-1]
        self.eigval = self.eigval[new_id]
        self.V = self.V[:,new_id]
        
        self.eigval = self.eigval.real[:k]
        self.V = self.V.real[:,:k]
        
        if diagonal:
            self.deltaA = np.diag(self.perturbation((k,),seed=seed))
        else:
            self.deltaA = self.perturbation((k,k),sigma=self.sigma, mu = self.mu ,seed=seed)
            
        if symmetric:
            self.deltaA = self.deltaA.T.dot(self.deltaA)
            
        eigval,V = self.eigval.real, self.V.real
        #V.dot((np.diag(eigval)+self.deltaA).dot(V.T))
        return V.dot(self.deltaA.dot(V.T))
    
    def perturbation(self,shape,sigma=1,mu=0,seed=None):
        np.random.seed(seed=seed)
        return sigma*np.random.randn(*shape) + mu
        
        
class CyclicPerturbation():
    def __init__(self,Krd,Mrd,selection_operetor,nsectors,dimension,frd=None,perturbation_order=10,sigma=1,mu=0,symmetric=True,diagonal=False):
        
        self.Krd = Krd
        self.Mrd = Mrd
        self.theta = (2.0*np.pi)/nsectors
        self.nsectors = nsectors
        self.perturbation_order = perturbation_order
        self.symmetric = symmetric
        self.diagonal = diagonal
        
        if callable(sigma):
            self.sigma = sigma
        else:
            self.sigma = lambda i : sigma

        if not callable(mu):
            self.mu = lambda i : mu
        else:
            self.mu = mu

        self.dimension = dimension
        self.selection_operetor = selection_operetor
        self.ndofs = Krd.shape[0]
        if callable(frd):
            self.frd  = frd
        else:
            self.frd = lambda i : np.zeros(self.ndofs)
    
    def create_cyclic_perturbed_matrices(self,seed_index=0, perturbation_order=None, sigma=None,mu=None,symmetric=None,diagonal=None):
    
        if perturbation_order is None:
            perturbation_order = self.perturbation_order
            
        if sigma is None:
            sigma = self.sigma

        if mu is None:
            mu = self.mu
            
        if symmetric is None:
            symmetric = self.symmetric
            
        if diagonal is None:
            diagonal = self.diagonal
            
        # creating eigenbases for perturbation
        Op = LinearSys(self.Krd,self.Mrd)
        D = Op.getLinearOperator()
        eigval, V = sparse.linalg.eigs(D,k=perturbation_order)

        # creating pertubation list based on sector index
        delta_list = []
        for i in range(self.nsectors):
            delta_list.append(eigvec_perturbation(V.real,seed=seed_index+i,sigma=sigma(i),mu=mu,symmetric=symmetric,diagonal=diagonal))

        return delta_list
     
    def create_interface_pair_dict(self,nsectors=None):
        if nsectors is None:
            nsectors = self.nsectors

        pair_dict = {}
        for i in range(nsectors):
            iplus = i + 1
            iminus = i - 1
            if iminus==-1:
                iminus = nsectors  - 1
            if iplus==nsectors:
                iplus = 0

            pair_dict[i] = {'Left':(i,iplus),'Right':(i,iminus)}

        return pair_dict

    def create_cyclic_perturbed_system(self,seed_index=0,stiffness=True,mass=False,interface=False, sigma=None,mu=None):

        if not callable(sigma):
            self.sigma = lambda i : sigma
        if not callable(mu):
            self.mu = lambda i : mu

        sred = self.selection_operetor
        B1 = sred.build_B('Left')
        B2 = sred.build_B('Right')
        pair_dict = self.create_interface_pair_dict()

        if stiffness:
            K_delta_list = self.create_cyclic_perturbed_matrices(seed_index,sigma=self.sigma,mu=mu)
        if mass:
            M_delta_list = self.create_cyclic_perturbed_matrices(seed_index,sigma=self.sigma,mu=mu)

        apply_rotation = lambda A,R : R.T.dot(A.dot(R))
        K_dict = {}
        M_dict = {}
        B_dict = {}
        f_dict = {}
        for i in range(self.nsectors):
            Ri = create_voigt_rotation_matrix(self.ndofs,i*self.theta,dim=self.dimension)
            if stiffness:
                Ki = apply_rotation(self.Krd + K_delta_list[i],Ri)
            else:
                Ki = apply_rotation(self.Krd,Ri)

            if mass: 
                Mi = apply_rotation(self.Mrd + M_delta_list[i],Ri)
            else:
                Mi = apply_rotation(self.Mrd,Ri)

            K_dict[i] = Ki
            M_dict[i] = Mi
           
                
            if interface:
                raise('Not supported')
            else:
                B_dict[i] = {pair_dict[i]['Left']:B1,pair_dict[i]['Right']:B2}
            
            f_dict[i] = self.frd(i)
                    
        return K_dict, M_dict, B_dict, f_dict
            
    