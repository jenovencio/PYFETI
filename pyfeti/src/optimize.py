from scipy import optimize
from scipy import sparse
import numpy as np
from unittest import TestCase, main
import logging
from pyfeti.src.feti_solver import SolverManager, SerialFETIsolver

def newton(f,Df,x0,epsilon=1.e-8,max_iter=30):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : np.array or LinearSolver
        Derivative of f(x).
    x0 : np.array
        Initial guess for a solution f(x)=0.
    epsilon : float
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : int
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            delta_x = np.linalg.solve(Dfxn,fxn)
            x = xn - dalta_x
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    sol = optimize.OptimizeResult()
    sol.success = False
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if np.linalg.norm(fxn) < epsilon:
            
            sol.x = xn
            sol.fun = fxn
            sol.nit = n
            sol.success = True
            return sol

        try:
            if isinstance(Df,np.ndarray):
                delta_x = np.linalg.solve(Df,-fxn)
            
            elif isinstance(Df,LinearSolver):
                delta_x = Df.solve(-fxn)
            else:
                raise TypeError('Only np.arrays and LinearSolver are supported for the Jacobian')
            xn = xn + delta_x
        except:
            logging.error('Jacobian is not positive definite. No solution found.')
            return sol
        
    logging.warning('Exceeded maximum iterations. No solution found.')
    return sol

class LinearSolver():
    ''' Provides and interface for 
    Linear Solvers 
    e.g 

    solve the system A.dot(x) = b using cg

    >>>A = np.array([[2.,-1.],[-1.,1.]])
    >>>r = np.array([1.,1.])
    >>>solverA = lambda A : lambda x : sparse.linalg.cg(A,x)[0]
    >>>solver = solverA(A)
    >>>ls = LinearSolver(solver)
    >>>x = ls.solve(r)
    >>>print(x)
    array([2., 3.])
    
    '''
    def __init__(self,solver=None,options={},**kwargs):

        self.solver = solver
        self._local_problems = {}
        self._map = {}
        self.__dict__.update(options)
        self.__dict__.update(kwargs)


    @property
    def options(self):
        return self.__dict__
    
    @options.setter
    def options(self,options):
        self.__dict__.update(options)

    @property
    def map(self):
        return self._map
    
    @map.setter
    def map(self,map_dict):
        self._map = map_dict

    def map_array2dict(self,v):
        v_dict = {}
        for global_index, local_info in self.map.items():
            for interface_id in local_info:
                v_dict[interface_id] = v[np.ix_(global_index)]

        return v_dict

    def map_dict2array(self,v_dict):
        v_dict = {}
        for global_index, local_info in self.map.items():
            for interface_id in local_info:
                v[np.ix_(global_index)] = v_dict[interface_id]

        return v_dict

    @property
    def local_problem(self):
        return self._local_problems

    @local_problem.setter
    def local_problem(self,local_problems_dict):
        self._local_problems = local_problems_dict

    def solve(self,r):
        return self.solver(r)


class FETI(LinearSolver):
    def __init__(self,solver=None):
        super().__init__(solver)
        
    def solve(self,r):
        ''' 
        F is linearized arround v ('which is lambda'), and solution u from
        the local equilibrium

        '''
        r_dict = self.map_array2dict(r)
        v_dict = self.map_array2dict(self.v)
        # transforming the nonlinear problem and linear problem
        local_problem_dict = {}
        for key, nonlinear_obj in self.local_problem.items():
                local_problem_dict[key] = nonlinear_obj.build_linear_problem(r_dict,self.w0)

        # instanciating a FETIsolver
        solver = SerialFETIsolver({},{},{})
        for key, local_problem in local_problem_dict.items():
            solver.manager.add_localproblem(local_problem,local_id=key)
    
        solution = solver.solve()

        return solution.interface_lambda




def feti(localproblem_dict,**kwargs):
    ''' Recieves a dict of local problems and 
    solve the dual interface return lambda

    '''
    solver = SerialFETIsolver({},{},{})
    for key, local_problem in localproblem_dict.items():
        solver.manager.add_localproblem(local_problem,local_id=key)
    
    solution = solver.solve()

    return solution.interface_lambda

class Test_Optimize(TestCase):
    def test_linearsolver(self):
        '''solve the system A.dot(x) = b using cg
        '''

    A = np.array([[2.,-1.],[-1.,1.]])
    r = np.array([1.,1.])
    x_target = np.linalg.solve(A,r)

    solverA = lambda A : lambda x : sparse.linalg.cg(A,x)[0]
    solver = solverA(A)

    
    ls = LinearSolver(solver)
    x = ls.solve(r)
    
    np.testing.assert_array_almost_equal(x,x_target,decimal=10)


if __name__=='__main__':
    main()
