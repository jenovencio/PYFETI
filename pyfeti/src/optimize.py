from scipy import optimize
from scipy import sparse
import numpy as np
from unittest import TestCase, main
import logging
from pyfeti.src.feti_solver import SolverManager, SerialFETIsolver
from pyfeti.src import optimize as opt
import numdifftools as nd

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
    norm_delta_x = 1.e2*epsilon
    xn = x0
    for n in range(0,max_iter):
        try:
            fxn = f(xn)
        except:
            logging.error('Error evaluation the residual function. No solution found.')
            sol.nit = n
            return sol

        norm_f = np.linalg.norm(fxn)
        print('norm F = %2.4e' %norm_f)
        if (norm_f) < epsilon or (norm_delta_x < epsilon):
            
            sol.fun = fxn
            sol.success = True
            return sol

        try:
            if isinstance(Df,np.ndarray):
                delta_x = np.linalg.solve(Df,-fxn)
            
            elif isinstance(Df,LinearSolver):
                delta_x = Df.solve(xn,-fxn)

            elif callable(Df):
                delta_x = np.linalg.solve(Df(xn),-fxn)
            else:
                raise TypeError('Only np.arrays and LinearSolver are supported for the Jacobian')
            norm_delta_x = np.linalg.norm(delta_x)
            xn = xn + delta_x
            sol.x = xn
            sol.nit = n

        except:
            logging.error('Jacobian is not positive definite. No solution found.')
            return sol
        
    logging.warning('Exceeded maximum iterations. No solution found.')
    return sol


def newton_krylov_cont(fun,x0,p_range,p0=None, jacx=None, jacp=None ,step=1.0,max_int=500,max_int_corr=50,tol=1.0E-6,max_dp=1.0,
                       correction_method='cg',print_mode=False):
    ''' This function applies a continuation technique
    in the fun(x,p) which 

    Parameters:
    ------------
        fun : callable
            callable function with two parameters x and p
        x0 : np.array
            initial guess for the initial point
        p_range : tuple 
            tuple with lower and upper limit for the p parameter

    
    '''

    y_list = [] # store coverged points
    p_list = [] # store coverged parameters
    info_dict = {'success' : False} # store algorithm infos

    if not isinstance(x0, np.ndarray):
        x0 = np.array([x0])

    if p0 is None:
        p0 = p_range[0]

    if not isinstance(p0, np.ndarray):
        p0 = np.array([p0])

    x_size = len(x0)
    p_size = len(p0)

    if p_size>1:
        raise NotImplementedError('Not implemented!')

    fx = lambda p : lambda x : fun(x,p)
    fp = lambda x : lambda p : fun(x,p)

    if jacp==None:
        jacp = lambda x, p : nd.Jacobian(fp(x),n=1)(p)

    if jacx==None:        
        jacx = lambda x, p : nd.Jacobian(fx(p),n=1)(x)

    krylov_solver = getattr( sparse.linalg,correction_method)
    



    J_inv_func = lambda p : lambda x, r : krylov_solver(jacx(x,p),r)[0]
    Jinv = lambda p : opt.LinearSolver(solver = J_inv_func(p))
    sol = lambda xn, pn : newton(fx(pn),Jinv(pn),xn,epsilon=tol,max_iter=max_int_corr)
    newton_sol = sol(x0,p0)
    
    #
    #jac = lambda x_aug : linalg.block_diag(*([Gy(x_aug[:n],x_aug[n:])]*2))

    if newton_sol.success:
        xn,pn = newton_sol.x,p0
    else:
        raise ValueError('Initial solution did not converge.')

    b = jacp(xn,pn)
    A = jacx(xn,pn)
    sub_kernel = krylov_solver(A,-b)[0]
    tn = np.concatenate((sub_kernel,np.array([1.0])))
    tn /= np.linalg.norm(tn)

    #tn = np.zeros(x_size+p_size,dtype=np.complex)
    #tn[-1] = 1.0
    #t0 = tn[:]
    x_aug_n = np.concatenate((xn,pn))
    par2aug =  lambda x_aug : (x_aug[:x_size],x_aug[x_size:x_size+p_size])
    
    #R_aug = lambda x_aug : np.vstack(( fun(*par2aug(x_aug)),
    #                                   tn.dot(x_aug.conj())))

    real_p = lambda x_aug : np.abs(x_aug[-1].imag)/np.abs(x_aug[-1].real)
    R_aug = lambda x_aug : np.concatenate( (fun(*par2aug(x_aug)),
                                       np.array([tn.conj().dot(x_aug) + real_p(x_aug)  ])))

    C_aug = lambda x_aug, tn : np.block([np.block([jacx(*par2aug(x_aug_n)),jacp(*par2aug(x_aug_n))]).dot(tn),np.array([[0.0]])]).T

    Jac_aug = lambda x_aug : np.block( [[ jacx(*par2aug(x_aug)),jacp(*par2aug(x_aug))],[tn.conj()]])

    J_num =lambda x_aug : nd.Jacobian(R_aug,n=1)(x_aug)
                                     
    Df = lambda x_aug, tn :  Jac_aug(x_aug)

    r = lambda x_aug, tn : np.block([R_aug(x_aug),C_aug(x_aug, tn)])

    J_inv_func = lambda x_aug, tn, r : krylov_solver(Jac_aug(x_aug),r)[0]

    Jinv = opt.LinearSolver(solver = J_inv_func)

    pn += step*tn[-1].real
    xn += step*tn[:-1]
    x_aug_n = np.concatenate((xn,pn))
    for i in range(max_int):
        
        newton_sol = newton(R_aug,J_num,x_aug_n,epsilon=1.0e-12,max_iter=20)
        #newton_sol = newton(r,Df,(x_aug_n,tn),epsilon=tol,max_iter=2)

        x_aug_n = newton_sol.x
        xn, pn = par2aug(x_aug_n)
        pn = pn.real

        if newton_sol.success:
            
            y_list.append(xn)
            p_list.append(pn[0])

        

        x_aug_n = np.concatenate((xn,pn))   
        
        # find tangent dx/dp
        #b = Jac_aug(x_aug_n).dot(tn)
        #b[-1] = 0.0

        #delta_tn = Jinv.solve(x_aug_n,-b)
        #tn += delta_tn

        #k = Jinv.solve(x_aug_n,t0)
        tn = np.linalg.solve(Jac_aug(x_aug_n),tn)


        #b = jacp(xn,pn)
        #b += Jac_aug(x_aug_n).dot(tn)[:-1]
        #A = jacx(xn,pn)
        #sub_kernel = krylov_solver(A,-b)[0]
        #sub_kernel /= np.linalg.norm(sub_kernel)
        #kernel = np.concatenate((sub_kernel,np.array([1.0])))

        if np.abs(Jac_aug(x_aug_n).dot(tn)[0])>tol:
            print('Kernel not orthogonal to the Jacobian')
            #raise ValueError('Not orthogonal')

        tn /= np.linalg.norm(tn)
        #kernel /= np.linalg.norm(kernel)

        #tn = np.dot(tn,kernel)*kernel
        #tn /= np.linalg.norm(tn)
        #tn *= step

        #update parameters
        
        if step*tn[-1].real>3*step:
            pn += step
            xn = xn
        else:
            pn += step*tn[-1].real
            xn += step*tn[:-1]
        x_aug_n = np.concatenate((xn,pn))

        if np.real(tn[-1])<=0.0:
            stopp = 1

         

        if not (p_range[0]<=pn[0]<=p_range[1]):
            break

    return np.array(y_list).T, np.array(p_list), info_dict





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

    def solve(self,*args):
        return self.solver(*args)


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
