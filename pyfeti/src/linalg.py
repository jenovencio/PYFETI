# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:22:46 2017

@author: Guilherme Jenovencio


This module is the Applied Mechanical Numerical Algebra Library (AMNA)

This library intend to solve linear problems and non-linear problem filling the 
gaps of where standards libraries like numpy, scipy cannot handle efficiently,
for example, solving singular system

methods:
    cholsps: Cholesky decomposition for Symmetry Positive Definite Matrix

"""

import logging 
from unittest import TestCase, main
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix, issparse, lil_matrix, linalg as sla
from scipy import linalg
from scipy.sparse.linalg import LinearOperator

import sys
sys.path.append('../..')
from pyfeti.src.utils import OrderedSet, Get_dofs, save_object, MapDofs


def cholsps(A, tol=1.0e-8):    
    ''' This method return the upper traingular matrix of cholesky decomposition of A.
    This function works for positive semi-definite matrix. 
    This functions also return the null space of the matrix A.
    Input:
    
        A -> positive semi-definite matrix
        tol -> tolerance for small pivots
        
    Ouputs:
        U -> upper triangule of Cholesky decomposition
        idp -> list of non-zero pivot rows
        R -> matrix with bases of the null space
        
    '''
    [n,m] = np.shape(A)
    
    if n!=m:
        print('Matrix is not square')
        return
    
    
    L = np.zeros([n,n])
    #L = sparse.lil_matrix((n,n),dtype=float) 
    #A = sparse.csr_matrix(A)
    idp = [] # id of non-zero pivot columns
    idf = [] # id of zero pivot columns
    
    if issparse(A):
        Atrace = np.trace(A.A)
        A = A.todense()
    else:    
        Atrace = np.trace(A)
        
    tolA = tol*Atrace/n
    
    for i in range(n):
        Li = L[i,:]
        Lii = A[i,i] - np.dot(Li,Li)
        if Lii>tolA:
            L[i,i] = np.sqrt(Lii)
            idp.append(i)
        elif abs(Lii)<tolA:
            L[i,i] = 0.0
            idf.append(i)
    
        elif Lii<-tolA:
            logging.debug('Matrix is not positive semi-definite.' + \
                          'Given tolerance = %2.5e' %tol)
            return L, [], None
    
        for j in range(i+1,n):
            if L[i, i]>tolA:
                L[j, i] = (A[j, i] - np.dot(L[i,:],L[j,:]))/L[i, i]
            
            
    # finding the null space
    rank = len(idp)
    rank_null = n - rank
    
    U = L.T 
    R = None
    if rank_null>0:
        Im = np.eye(rank_null)
        
        # Applying permutation to get an echelon form
        
        PA = np.zeros([n,n])
        PA[:rank,:] = U[idp,:]
        PA[rank:,:] = U[idf,:]
        
        # creating block matrix
        A11 = np.zeros([rank,rank])
        A12 = np.zeros([rank,rank_null])
        
        A11 = PA[:rank,idp]
        A12 = PA[:rank,idf]
        
        
        R11 = np.zeros([rank,rank_null])
        R = np.zeros([n,rank_null])
        
        # backward substitution
        for i in range(rank_null):
            for j in range(rank-1,-1,-1):
                if j==rank-1:
                    R11[j,i] = -A12[j,i]/A11[j,j]
                else:
                    R11[j,i] = (-A12[j,i] - np.dot(R11[j+1:rank,i],A11[j,j+1:rank]) )/A11[j,j]
                
        # back to the original bases
        R[idf,:] = Im
        R[idp,:] = R11
        
        logging.debug('Null space size = %i' %len(idf))
            
    return U, idf, R   

def splusps(A,tol=1.0e-6):
    ''' This method return the upper traingular matrix based on superLU of A.
    This function works for positive semi-definite matrix. 
    This functions also return the null space of the matrix A.
    Input:
    
        A -> positive semi-definite matrix
        tol -> tolerance for small pivots
        
    Ouputs:
        U -> upper triangule of Cholesky decomposition
        idp -> list of non-zero pivot rows
        R -> matrix with bases of the null space
    '''
    [n,m] = np.shape(A)
    
    if n!=m:
        print('Matrix is not square')
        return
    
    
    idp = [] # id of non-zero pivot columns
    idf = [] # id of zero pivot columns
    
    if not isinstance(A,csc_matrix):  
        A = csc_matrix(A)

    lu = sla.splu(A)

    #L = lu.L
    U = lu.U
    Pr = csc_matrix((n, n))
    Pc = csc_matrix((n, n))
    Pc[np.arange(n), lu.perm_c] = 1
    Pr[lu.perm_r, np.arange(n)] = 1

    #L1 = (Pr.T * L).A
    #L2 = (U*Pc.T).A

    Utrace = np.trace(U.A)

    diag_U = np.diag(U.A)/Utrace

    idf = np.where(abs(diag_U)<tol)[0].tolist()
    
    if len(idf)>0:
        R = calc_null_space_of_upper_trig_matrix(U,idf)
        R = Pc.A.dot(R)
    else:
        R = np.array([])

    #for v in R.T:
    #    is_null_space(A,v, tol)

    return  lu, idf, R

def calc_null_space_of_upper_trig_matrix(U,idf=None):
    ''' This function computer the Null space of
    a Upper Triangule matrix which is can be a singular
    matrix

    argument
        U : np.matrix
            Upper triangular matrix
        idf: list
            index to fixed if the matrix is singular
    
    return
        R : np.matrix
            null space of U
    
    '''

    
    # finding the null space
    n,n = U.shape
    rank_null =len(idf)
    rank = n - rank_null
    
    U[np.ix_(idf),np.ix_(idf)] = 0

    # finding the null space
    idp = set(range(n))
    for fid in idf:
        idp.remove(fid)
    
    idp = list(idp)

    R = None
    if rank_null>0:
        Im = np.eye(rank_null)
        
        # Applying permutation to get an echelon form
        
        PA = np.zeros([n,n])
        PA[:rank,:] = U.A[idp,:]
        PA[rank:,:] = U.A[idf,:]
        
        # creating block matrix
        A11 = np.zeros([rank,rank])
        A12 = np.zeros([rank,rank_null])
        
        A11 = PA[:rank,idp]
        A12 = PA[:rank,idf]
        
        
        R11 = np.zeros([rank,rank_null])
        R = np.zeros([n,rank_null])
        
        # backward substitution
        for i in range(rank_null):
            for j in range(rank-1,-1,-1):
                if j==rank-1:
                    R11[j,i] = -A12[j,i]/A11[j,j]
                else:
                    R11[j,i] = (-A12[j,i] - np.dot(R11[j+1:rank,i],A11[j,j+1:rank]) )/A11[j,j]
                
        # back to the original bases
        R[idf,:] = Im
        R[idp,:] = R11

        return R

def pinv_and_null_space_svd(K,tol=1.0E-8):
    ''' calc pseudo inverve and
    null space using SVD technique
    '''

    if issparse(K):
        K = K.todense()
        
    n, n = K.shape
    V,val,U = np.linalg.svd(K)
        
    total_var = np.sum(val)
        
    norm_eigval = val/val[0]
    idx = [i for i,val in enumerate(norm_eigval) if val>tol]
    val = val[idx]
        
        
    invval = 1.0/val[idx]

    subV = V[:,idx]
        
    Kinv =  np.matmul( subV,np.matmul(np.diag(invval),subV.T))
        
    last_idx = idx[-1]
    if n>len(idx):
        R = np.array(V[:,last_idx+1:])
    else:
        R = np.array([])

    return Kinv,R
        
def is_null_space(K,v, tol=1.0E-3):
    ''' this function checks if 
    a vector is belongs to the null space of
    K matrix

    argument:
    K : np.matrix
        matrix to check the kernel vector
    v : np.array
        vector to be tested
    tol : float
        tolerance for the null space

       '''

    norm_v = np.linalg.norm(v)
    r = K.dot(v)
    norm_r = np.linalg.norm(r)

    ratio = norm_r/norm_v

    if ratio<=tol:
        return True
    else:
        return False


class LinearSys():
    def __init__(self,A,M,alg='splu'):
        self.A = A
        self.M = M
        self.ndof = self.A.shape[0]
        self.alg =alg
        self.lu = None
        if self.alg=='splu':
            self.lu = sparse.linalg.splu(A)
        
    def solve(self,b):
        A = self.A
        M = self.M
        b = np.array(b)
        b_prime = np.array(M.dot(b)).flatten()
        if self.alg=='splu':
            x = self.lu.solve(b_prime)
        elif self.alg=='cg':
            x = sparse.linalg.cg(A,b_prime)[0]
        else:
            raise('Algorithm &s not supported' %self.alg)
            
        return x
        
    def normM(self,b):
        M = self.M
        b_prime = np.array(M.dot(b)).flatten()
        return b.dot(b_prime)    
        
    def getLinearOperator(self):
        return LinearOperator((self.ndof,self.ndof), matvec=self.solve)  
        
class ProjLinearSys():      
    def __init__(self,A,M,P,precond=None,linear_solver=None,solver_tol=1.0E-10):
        self.A = A
        self.M = M
        self.P = P
        self.precond = precond
        self.solver_counter = 0
        self.num_iters=0
        self.linear_solver = linear_solver
        self.solver_tol = solver_tol
        self.Ap = P.conj().T.dot(A.dot(P))
        

    def solve(self,b):
        M = self.M
        P = self.P
        b = np.array(b)
        self.solver_counter += 1
        b_prime =  np.array(P.conj().T.dot(M.dot(P.dot(b)))).flatten()
        Ap = self.Ap
        if self.linear_solver==None:
            return sparse.linalg.cg(Ap,b_prime,M = self.precond, callback=self.counter, tol=self.solver_tol)[0]
        else:
            return self.linear_solver(Ap,b_prime,M = self.precond, callback=self.counter)[0]
        
    
    def counter(self,xk):
        ''' count number of iterations
        '''
        self.num_iters+=1
        #print(self.num_iters)
        
    
    def normM(self,b):
        M = self.M
        b_prime = np.array(M.dot(b)).flatten()
        return b.dot(b_prime)    
        
    def getLinearOperator(self):
        ndof = self.A.shape[0]
        return LinearOperator((ndof,ndof), matvec=self.solve , dtype=np.complex)  
    
class ProjectorOperator(LinearOperator):    
    ''' This interface provides a interface for the Projected Operator
    such that

    u = P A P v

    '''
    def __init__(self,A,P,dtype=np.float,shape=None):
        super().__init__(dtype=dtype,shape=shape)
        
        try:
            self.PAP = P.H.dot(A.dot(P))
        except:
            self.PAP = P.T.dot(A.dot(P))

    def _matvec(self,v):
        return self.PAP.dot(v)
        


class DualLinearSys():      
    def __init__(self,A,B,nc,sigma=0.0,precond=None, projection=None):
        ''' Creates a linear operator such as
        
        A = [K C*^T]
            [C   0 ]
            
        B = [M  0]
            [0  0]
            
        [xk, lambda]^T = A^-1 M
        
        where lambda has size nc
        '''
        self.A = A
        self.B = B
        self.nc = nc
        self.precond = precond
        self.num_iters=0
        self.ndofs = A.shape[0]
        self.u_dofs = self.ndofs - nc
        self.sigma = sigma
        
        
        

        
        if projection is None:
            self.P = sparse.eye(self.u_dofs )
           
        else:
            self.P = projection
           
        self.M = self.P.conj().T.dot(B[:self.u_dofs,:self.u_dofs]).dot(self.P)
        self.K = self.P.conj().T.dot(A[:self.u_dofs,:self.u_dofs]).dot(self.P)
        self.lu = sparse.linalg.splu(self.K - self.sigma*self.M)
        #lu = sparse.linalg.splu(self.K - sigma*self.M)
        #self.K_inv = lu.solve
        self.K_inv = LinearOperator((self.u_dofs,self.u_dofs), matvec = self.lu.solve)
        self.C = A[self.u_dofs:,:self.u_dofs]
        #self.F = lambda b : self.C.dot(self.K_inv(self.C.conj().T.dot(b)))
        #self.F LinearOperator((ndof,ndof), matvec = lambda b : self.C.dot(self.K_inv(self.C.conj().T.dot(b)))) 
        self.F = LinearOperator((nc,nc), matvec = self.F_operator)
            
    def F_operator(self,b):
    
        return self.C.dot(self.K_inv(self.C.conj().T.dot(b))) 
    
    
    def solve(self,b):
        A = self.A
        B = self.B
        
        M = self.M 
        u = self.P.dot(b[:self.u_dofs])
        u_prime = self.K_inv.dot((M.dot(u)))
        lambda_n1 = sparse.linalg.cg(self.F,self.C.dot(u_prime), M = self.precond, callback=self.counter, tol=1e-16)[0]
        
        u_n1 = self.P.dot(u_prime - self.K_inv(self.C.conj().T.dot(lambda_n1)))
        return np.concatenate((u_n1,lambda_n1))
    
    def counter(self,xk):
        ''' count number of iterations
        '''
        self.num_iters+=1
        #print(self.num_iters)
        
    def normM(self,b):
        B = self.B
        b_prime = np.array(B.dot(b)).flatten()
        return b.dot(b_prime)    
        
    def getLinearOperator(self):
        ndof = self.A.shape[0]
        return LinearOperator((ndof,ndof), matvec=self.solve) 
  
class ProjPrecondLinearSys():      
    def __init__(self,A,P, incomplete=False,drop_tol=None, fill_factor=None):
        
        self.A = A
        self.P = P
        ndof = A.shape[0]
        self.solver_counter = 0
        if incomplete:
            lu = sparse.linalg.spilu(A, drop_tol=drop_tol, fill_factor=fill_factor)
        else:
            lu = sparse.linalg.splu(A)
            
        self.lu = lu
        self.A_inv = LinearOperator((ndof,ndof), matvec=lu.solve, dtype = P.dtype) 
        
    def solve(self,b):
        
        P = self.P
        if b.dtype=='complex':
            A_inv = self.A_inv
            u_real = A_inv.dot( (P.dot(b)).real)
            u_imag = A_inv.dot( (P.dot(b)).imag)
            u = np.zeros(u_real.shape, dtype=np.complex)
            u.real = u_real
            u.astype(np.complex)
            u.imag = u_imag
            
        else:    
            A_inv = self.A_inv
            u = A_inv.dot( (P.dot(b)))

        self.solver_counter += 1
        return P.dot(u)
        
    def getLinearOperator(self):
        ndof = self.A.shape[0]
        return LinearOperator((ndof,ndof), matvec=self.solve, dtype=self.P.dtype)  

def create_permutation_matrix(row_indexes,col_indexes,shape):
    ''' create a Permutation matrix based on local id
    
    '''
    P = lil_matrix(shape, dtype=np.int8)
    P[row_indexes, col_indexes] = 1
    return P.toarray()

def map_matrix(map_dofs,**kargs):
    map_obj = MapDofs(map_dofs,**kargs)
    total_dof_length = map_obj.local_dofs_length()
    global_dof_lenth = map_obj.global_dofs_length()
    dof_list = np.arange(total_dof_length)
    
    L = np.zeros((global_dof_lenth,total_dof_length)) 
    domain_list = list(map_obj.get_local_map_dict)
    for domain_id in domain_list:
        local_rows = map_obj.get_domain_rows(domain_id)
        global_rows = map_obj.global_dofs(domain_id)
        L += create_permutation_matrix(global_rows, local_rows, (global_dof_lenth,total_dof_length))

    return L

def elimination_matrix_from_map_dofs(map_dofs,**kargs):

    L = map_matrix(map_dofs,**kargs)

    for i,row in enumerate(L):
        scale = sum(row)
        if scale>1.0:
            L[i,:] = (1.0/scale)*L[i,:]
    return L

def expansion_matrix_from_map_dofs(map_dofs,**kargs):

    L = map_matrix(map_dofs,**kargs)
 
    return L.T

def get_unit_rotation_matrix(alpha_rad,dim=3,axis='z'):  
    ''' Create a unitary rotation matrix based on a angle alpha 
    and an axis of rotation

    Parameters:
        alpha_rad : float
            angle in rad 
        dim : int
            2 or 2D rotatuon 3 for 3D rotation matrix
        axis : str
            'x', 'y' or 'z'

    '''
    

    
    if dim==3 and not (axis in 'xyz'):
        raise('Axis %s is not supperted. Please select x,y or z.' %axis)

    #defining anti-clock wise rotation
    alpha_rad = -alpha_rad

    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)

    if dim==3 and axis=='z':
        R_i = np.array([[cos_a,-sin_a,0.0],
                        [sin_a, cos_a,0],
                        [0.0, 0.0, 1.0]])
    
    elif dim==3 and axis=='x':    
        R_i = np.array([[0.0,cos_a,-sin_a],
                        [0.0, sin_a, cos_a],
                        [1.0, 0.0, 0.0]])

    elif dim==3 and axis=='y':    
        R_i = np.array([[cos_a,0.0,-sin_a],
                        [sin_a, 0.0, cos_a],
                        [0.0, 1.0, 0.0]])

    elif dim==2:
        R_i = np.array([[cos_a, -sin_a],
                       [sin_a, cos_a]])    
    else:
        raise('Dimension not supported')      
        
    return R_i

def find_cyclic_node_pairs(node_set_left,node_set_right,angle,node_coord,dim=2,tol_dist=1.0E-6 ):
    
    R = get_unit_rotation_matrix(angle,dim=dim)
    node_pair_dict = {}
    node_set_right_copy = node_set_right[:]
    for node_id_1 in node_set_left:
        min_dist = 1.0E8
        coord_1 = node_coord[node_id_1]
        for node_id_2 in node_set_right_copy:
            coord_2 = node_coord[node_id_2]
            dist = np.linalg.norm(coord_1 - R.T.dot(coord_2))
            if dist<min_dist:
                min_dist = dist
                node_pair = node_id_2
                
        
        if  min_dist>=tol_dist:
            raise('Could not find node pairs given the tolerance = %e' %tol_dist)
        else:
            # update dict
            node_pair_dict[node_id_1] = node_pair

        try:

            # remove nodes to speed-up search
            node_set_right_copy.remove(node_pair_dict[node_id_1])
        except:
            pass

    new_node_set_right = []
    for key in node_set_left:
        new_node_set_right.append(node_pair_dict[key])
        
    return new_node_set_right

def create_voigt_rotation_matrix(n_dofs,alpha_rad, dim=2, axis='z',unit='rad', sparse_matrix = True):
    ''' This function creates voigt rotation matrix, which is a block
    rotation which can be applied to a voigt displacement vector
    ''' 
    
    if n_dofs<=0:
        raise('Error!!! None dof was select to apply rotation.')
    
    if unit[0:3]=='deg':
        rotation = np.deg2rad(rotation)
        unit = 'rad'
        
    R_i = get_unit_rotation_matrix(alpha_rad,dim,axis)  
    
    n_blocks = int(n_dofs/dim)
    
    
    if n_blocks*dim != n_dofs:
        raise('Error!!! Rotation matrix is not matching with dimension and dofs.')
    if sparse_matrix:
        R = sparse.block_diag([R_i]*n_blocks)
    else:
        R = linalg.block_diag(*[R_i]*n_blocks)
    return R


class Pseudoinverse():
    ''' This class intend to solve singular systems
    build the null space of matrix operator and also 
    build the inverse matrix operator
    
    Ku = f
    
    where K is singular, then the general solution is
    
    u = K_pinvf + alpha*R
    
    argument
        K : np.array
            matrix to be inverted
        tol : float
            float tolerance for building the null space
        
    return:
        K_pinv : object
        object containg the null space and the inverse operator
    '''
    def __init__(self,method='splusps',tolerance=1.0E-8):
        
        self.list_of_solvers = ['cholsps','splusps','svd']
        if method not in self.list_of_solvers:
            raise('Selection method not avalible, please selection one in the following list :' %(self.list_of_solvers))

        self.solver_opt = method
        self.pinv = None
        self.null_space = np.array([])
        self.free_index = []
        self.tolerance = 1.0E-8
        self.matrix = None
    
    def set_tolerance(self,tol):
        ''' setting P_inverse tolerance
        
        arguments 
            tol : tol
                new pseudo-inverse tolerance
        return 
            None
        '''
        self.tolerance = tol
        
        return 
    
    def set_solver_opt(self,solver_opt):
        ''' This methods set the P_inverse method
        
            argument
                solver_opt : str
                    string with solver opt
            
            returns 
        '''
        if solver_opt in self.list_of_solvers:
            self.solver_opt = solver_opt
        else:
            raise('Error! Select solver is not implemented. ' + \
            '\n Please check list_of_solvers variable.')
        
    def compute(self,K,tol=None,solver_opt=None):
        ''' This method computes the kernel and inverse operator
        '''
        
        # store matrix to future use
        self.matrix = K
        
        if solver_opt is None:
            solver_opt = self.solver_opt

        if tol is None:
            tol = self.tolerance

        if solver_opt=='splusps':
            lu, idf, R = splusps(K,tol=tol)
            
            # add constraint in K matrix and applu SuperLu again
            if len(idf):
                Kmod = K[:,:] # creating copy because np.array is a reference
                idf_u = [np.argwhere(lu.perm_c==elem)[0][0] for elem in idf]
                idf_l = [np.argwhere(lu.perm_r==elem)[0][0] for elem in idf]
                Kmod[idf_u,:] = 0.0
                Kmod[:,idf_u] = 0.0
                Kmod[idf_u,idf_u] = K.diagonal().max()
                lu, idf_garbage, R_garbage = splusps(Kmod,tol=tol)
                idf = idf_u
                
            K_pinv = lu.solve
            
        elif solver_opt=='cholsps':
            U,idf,R =cholsps(K,tol=tol)
            U[idf,:] = 0.0
            U[:,idf] = 0.0
            U[idf,idf] = 1.0
            K_pinv = lambda f : linalg.cho_solve((U,False),f) 
            
        elif solver_opt=='svd':
            K_inv, R = pinv_and_null_space_svd(K,tol=tol)
            K_pinv = np.array(K_inv).dot
            idf = []
        
        else:
            raise('Solver %s not implement. Check list_of_solvers.')
        
        self.pinv = K_pinv
        self.free_index = idf
        if R is not None:
            self.null_space = R
        else:
            self.null_space = np.array([])
            
        return self
        
    def apply(self,f,alpha=np.array([]),check=False):
        ''' function to apply K_pinv
        and calculate a solution based on alpha
        by the default alpha is set to the zero vector
        
        argument  
            f : np.array
                right hand side of the equation 
            alpha : np.array
                combination of the kernel of K alpha*R
            check : boolean
                check if f is orthogonal to the null space
        '''
        K_pinv = self.pinv
        idf = self.free_index
        
        # f must be orthogonal to the null space R.T*f = 0 
        if idf:
            f[idf] = 0.0
        
        #if self.solver_opt == 'cholsps':
        #    f[idf] = 0.0
        
        if check:
            if not self.has_solution(f):
                raise('System has no solution because right hand side is \
                       \n not orthogonal to the null space of the matrix operator.')
        
        u_hat = K_pinv(f)
        
        if alpha.size>0:
            u_hat += self.calc_kernel_correction(alpha)
            
        return u_hat
        
    def calc_kernel_correction(self,alpha):
        ''' apply kernel correction to
        calculate another particular solution
        '''
        R = self.null_space
        u_corr = R.dot(alpha)
        return u_corr
    
    def check_null_space(self,tolerance=1.0E-3):
        ''' check null calculated null space is a null space 
        of self.matrix considering two aspects, 
        1. K*v = 0    where v is a vector in the R = [v1, v2  ...,vm]
            check ||K*v||/||v|| us < tolerance
        2. R is a full row rank matrix
        
        arguments:
            tolerance : float
                tolerance for the norm of the vector v in R
               by the the K1 matrix, which represents a tolerance for
               checking if v in R is really a kernel vector of K
        return
            bool : boolean
            
            True if all vector in null space are in the tolerance
        ''' 
        bool = False
        K = self.matrix
        R = self.null_space
        n,m = R.shape
        null_space_size =  0
        for v in R.T:
            if is_null_space(K,v, tol=tolerance):
                null_space_size += 1
        
        R_rank = np.linalg.matrix_rank(R.T)
        if m==null_space_size and R_rank==m:
            bool = True
        
        return bool
    
    def has_solution(self,f):
        ''' check if f is orthogonal to the null space
        
        arguments
            f : np.array
                right hand side of Ku=f
        return 
            boolean
        
        '''
        R = self.null_space
        v = R.T.dot(f)
        ratio = np.linalg.norm(v)/np.linalg.norm(f)
        
        bool = False
        if ratio<self.tolerance:
            bool= True
        return bool
        
        
class Matrix():
    '''  Basic matrix class 
    '''
    counter = 0

    def __init__(self,K,key_dict={},name=None,pseudoinverse_kargs={'method':'svd','tolerance':1.0E-8}):
        '''
        pseudoinverse_key_args=(method='splusps',tolerance=1.0E-8)
        '''
        Matrix.counter+=1
        self.id = Matrix.counter
        self.data = K
        self.key_dict = key_dict
        self.type = None
        self.issingular = None
        self.prefix = 'K'
        self.eliminated_id = set()
        self.psudeoinverve = Pseudoinverse(**pseudoinverse_kargs)
        self.inverse_computed = False
        if name is None:
            self.update_name()
        else:
            self.name = name
    
    def set_psudeoinverve_alg(self,name):
        ''' Parameters
                name : str
                    name of the pseudoinverse method
        '''
        pseudoinverse_key_args = {'method':name}
        self.psudeoinverve = Pseudoinverse(**pseudoinverse_key_args)
        self.issingular = None

    def update_name(self):
        self.name =  self.prefix  + str(self.id)

    @property
    def shape(self):
        return self.data.shape
    
    @property 
    def trace(self):
        return np.trace(self.data)
    
    @property 
    def det(self):
        return np.linalg.det(self.data)
    
    @property
    def eigenvalues(self):
        w, v = np.linalg.eig(self.data)
        return np.sort(w)[::-1]

    def dot(self,x):
        return K.dot(x)
        
    def inverse(self):
        pass
        
    @property
    def kernel(self):
        ''' compute the kernel of the matrix
        based on the pseudoinverse algorithm
        '''
        if not self.inverse_computed:
            self.psudeoinverve.compute(self.data)
            self.inverse_computed = True
            
        return self.psudeoinverve.null_space


    def apply_inverse(self, b):
        
        if not self.inverse_computed:
            self.psudeoinverve.compute(self.data)
            self.inverse_computed = True
    
        return self.psudeoinverve.pinv(b)
        
    def get_block(self,row_key,column_key):
        pass
     
    def eliminate_by_identity(self,dof_ids):
        ''' This function eliminates matrix rows and columns
        by replacing rows and columns by identity matrix
        
        [[k11, k12, k13                 [[k11, 0, k13
          k21, k22, k23]    ->            0, 1,    0]
          k21, k22, k23]                  k21, 0, k23]]

        Parameters:
            dof_ids : OrderedSet or a Str
                if OrderedSet a set of dofs to be eliminated by identity
                if string a key of self.key_dict which maps to the set of dof 
                to be eliminated

        return eliminated K matrix

        '''
        
        if isinstance(dof_ids,str):
            dofs = list(self.key_dict[dof_ids])
        else:
            dofs = list(dof_ids)
        
        if list(dofs)[0] is None:
            return 
 
        dirichlet_stiffness = self.trace/self.shape[0]       
        self.data[dofs,:] = 0.0
        self.data[:,dofs] = 0.0
        self.data[dofs,dofs] = dirichlet_stiffness
        self.eliminated_id.update(dofs)
        return self.data
        
    def save_to_file(self,filename=None):
        if filename is None:
            filename = self.name + '.pkl'
            print('Filename is = %s' %filename)
         
        save_object(self,filename)

    
class SparseMatrix(Matrix):
    '''  Basic matrix class 
    '''
    def __init__(self,K,key_dict={}):
        super().__init__(K,key_dict={})


class Vector():
    counter = 0

    def __init__(self,v,key_dict={},name=None):
        self.id = Vector.counter
        self.data = np.array(v)
        self.key_dict = key_dict
        self.prefix = 'v'
        
        if name is None:
            self.update_name()
        else:
            self.name = name
    
    def update_name(self):
        self.name =  self.prefix  + str(self.id)

    def replace_elements(self,dof_ids,value):
        '''
         Parameters:
            dof_ids : OrderedSet or a Str
                if OrderedSet a set of dofs will be replace by the value
                if string a key of self.key_dict which maps to the set of dof 
                will be replace by the value
            value : float
                float to replace the values in the initial array 
        
        return a new vnumpy.array
        '''
        if isinstance(dof_ids,str):
            dofs = list(self.key_dict[dof_ids])
        else:
            dofs = list(dof_ids)

        self.data[dofs] = value
        return self.data


class  Test_linalg(TestCase):
    def test_ProjectorOperator(self):

        A = 3*np.array([[2,-1,0],[-1,2,0],[0,-1,2]])
        P = np.array([[1,0,0],[0,1,0],[0,0,0]])
        PA = ProjectorOperator(A,P,shape=(3,3))
        b = np.array([-2,4,0])
        b1 = PA.dot(b)

        np.testing.assert_almost_equal(b1,P.dot(b1),decimal=10)

    def test_ProjectorOperator_with_minres(self):
        A = 3*np.array([[2,-1,0],[-1,2,0],[0,-1,2]])
        P = np.array([[1,0,0],[0,1,0],[0,0,0]])
        PA = ProjectorOperator(A,P,shape=(3,3))
        Asingular = P.dot(A.dot(P))
        b = np.array([-2,4,0])

        x_cg, info = sparse.linalg.cg(PA,b)
        x_minres, info = sparse.linalg.minres(PA,b)
        x_svd = (np.linalg.pinv(Asingular)).dot(b)

        np.testing.assert_almost_equal(x_cg,x_svd,decimal=10)
        np.testing.assert_almost_equal(x_minres,x_svd,decimal=10)

if __name__ == '__main__':
    main()