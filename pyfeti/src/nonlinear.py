

from pyfeti.src.nonlinalg import NonLinearOperator
from scipy import optimize
import numpy as np

class NonLinearLocalProblem():
    ''' This class provides a class to Nonlinear Local Problems, such that

    R(u,w0) = Z(u,w0) + f(u,w0) + sum B(i,j) lambda(ij) = 0

    for a given w0 array

    Parameters for instanciating class object

    Z : callback function or NonLinearOperator
        Z(u,w) where u is a state variable array and w is a parameters array
       
    f : callback function or NonLinearOperator
        f(u,w) where u is a state variable array and w is a parameters array

    B : dict
        dict with pair keys (i, j) selecting the elements at the interface

    length : int
        length of the array u in the callback Z(u,w)

    jac_dict : dict
        dict with the callback function Jacobian of Z in u 'JZu' and Jacobian of f in u
        where the callback must have the interface JZu(u,w) and Jfu(u,w)
        >>> {'JZu' : JZu, 'Jfu' : Jfu}.
        if not given a numerical approximation will be computed dependond on the chosen nonlinear solver

    solver_kargs : dict  Default = {'solver':'newton_krylov','method':'cgs', 'iter':None ,'tolerance':1.0E-8})
        dict with the nonlinear solver algorithm and its own internal parameters

    '''
    counter = 1
    def __init__(self,Z, B, f, length, jac_dict={'JZu' : None, 'Jfu' : None},
                 id=None,
                 solver_kargs={'method':'krylov','jac_options':{'method': 'cgs'}, 'iter':None ,'fatol':1.0E-8, 'atol':1.0E-8}):
        
        if isinstance(Z,NonLinearOperator):
            self.Z = Z
        else:
            self.Z = NonLinearOperator(Z)

        if isinstance(f,NonLinearOperator):
            self.f = f
        else:
            self.f = NonLinearOperator(f)

        self.B = B
        
        if id is None:
            self.id =  NonLinearLocalProblem.counter
        else:
            self.id = id

        self.length = length
        self.solver_kargs = solver_kargs
        self.alg_kwargs = self.solver_kargs.copy()
        del self.alg_kwargs['method']

        self.solution = None
        self.interface_size =  0
        self.neighbors_id = []
        self.crosspoints = {}
        
        NonLinearLocalProblem.counter+=1

    def solve(self,lambda_dict,w0,u_init=None):
        '''
        This method solve the following problem

         R(u,w0) = Z(u,w0) + f(u,w0) + sum B(i,j) lambda(ij) = 0

         given the solver_kargs parameters

        '''
        # force at the interface
        fb = np.zeros(self.length, dtype=np.complex)
        if lambda_dict is not None:
            # assemble interface force
            for interface_id, B in self.B.items():
                (local_id,nei_id) = interface_id
                if local_id>nei_id:
                    interface_id = (nei_id,local_id) 
                fb += B.T.dot(lambda_dict[interface_id])
    
        if u_init is None:
            u_init = np.zeros(self.length, dtype=np.complex)
        
        R = lambda u : self.Z.eval(u,w0) + self.f.eval(u,w0) + fb

        
        self.solution = optimize.root(R,u_init,method=self.solver_kargs['method'], options=self.alg_kwargs)

        return self.solution.x