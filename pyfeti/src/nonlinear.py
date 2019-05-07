

from pyfeti.src.nonlinalg import NonLinearOperator
from pyfeti.src.feti_solver import LocalProblem, SolverManager, SerialFETIsolver
from pyfeti.src import optimize as opt
from scipy import optimize, sparse
import numpy as np
import numdifftools as nd
import logging




class NonLinearLocalProblem(LocalProblem):
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
    
    def __init__(self,Z, B, f, length, jac_dict={'JZu' : None, 'Jfu' : None},
                 id=None,
                 solver_kargs={'method':'krylov','jac_options':{'method': 'cgs'}, 'maxiter':50 ,'fatol':1.0E-8, 'xatol':1.0E-8}):
        
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
            self.id =  list(B.keys())[0][0]
        else:
            self.id = id

        self.length = length
        self.solver_kargs = solver_kargs
        self.alg_kwargs = self.solver_kargs.copy()
        del self.alg_kwargs['method']
        self.dtype = np.complex
        self.solution = None
        self.u_init = np.zeros(self.length, dtype=np.complex)
        self.interface_size =  0
        self.neighbors_id = []
        self.crosspoints = {}
        
        #alias variables for LocalProblem methods, 
        # it should be the Linearized version of Z and f
        self.B_local = self.B
        self.K_local = self.Z
        self.f_local = self.f
        self.u_linear = None
        self.w0_linear = None
        
        self.get_neighbors_id()
 
    def residual(self,w0,fb=None):
        ''' return the residual function given
        a fixed parameter w0, and a fixed force at the
        interface fb.
        '''
        if fb is None:
            fb = np.zeros(self.length, dtype=self.dtype)
        Ru = lambda u : self.Z.eval(u,w0) + self.f.eval(u,w0) + fb
        n = fb.shape[0]
        return NonLinearOperator(Ru,shape=(n,n),dtype=self.dtype)

    def jacobian(self,w0,u=None,**kwargs):
        ''' Compute the Jacobian of the residual given u
        and the fixed paramenters w0
        '''
        if u is None:
            u = self.u_init

        JRu = nd.Jacobian(self.residual(w0),n=1)
        return JRu(u)

    def build_linear_problem(self,lambda_dict,w0,u=None,**kwargs):
        ''' Return a LinearProblem object based on w0
        and u
        ''' 

        flag = False
        if u is None:
            u = self.u_init        
            flag = True
        else:
            if u != self.u_linear or w0!=self.w0_linear:    
                flag = True
            else:
                raise Exception('u must be provided')

        if flag:
            self.K_local = self.jacobian(w0,u,**kwargs)
            self.u_linear = u
            self.w0_linear = w0

        fb = self.assemble_right_hand_side(lambda_dict) 
        f = self.residual(w0,fb).eval(u)
        
        return LocalProblem(self.K_local,self.B_local,f,self.id,**kwargs)

    def derivative_u_over_lambda(self,w0,u=None,**kwargs):
        ''' Return the implicit derivative of u over lambda
        B * du/dlambda
        '''

        if u is None:
            u = self.u_init

        JRu = self.jacobian(w0,u)
        derivative_dict = {}
        for key, B in self.B.items():
            derivative_dict[key] = B.dot(np.linalg.solve(JRu,-B.T))
        return derivative_dict

    def apply_local_F(self,lambda_dict,w0):
        local_F_dict = self.derivative_u_over_lambda(w0)
        local_u_dict = {}
        for key, B in self.B.items():
            local_id,nei_id = key
            if local_id>nei_id:
                l_key = (nei_id, local_id)
            else:
                l_key = key

            local_u_dict[key] = local_F_dict[key].dot(lambda_dict[l_key])
        return local_u_dict

    def solve_interface_displacement(self,lambda_dict,w0=None,u_init=None):
        '''
        This method solves the implicit displacement u at the interface
        such that u = f(lambda) where u is the solution of the following
        protblem

         R(u,w0) = Z(u,w0) + f(u,w0) + sum B(i,j) lambda(ij) = 0

         given the solver_kargs parameters

         The ui = B * u

        returns : dict
            a dictionary with keys based on interface pairs
            and values with displacement arrays 
        '''
        u = self.solve(lambda_dict,w0,u_init)
        return self.get_interface_dict(u)


    def assemble_right_hand_side(self,lambda_dict):
        ''' This function assembles the 
        sum B(i,j)*lambda(i,j)

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
        
        return fb

    def solve(self,lambda_dict,w0,u_init=None):
        '''
        This method solve the following problem

         R(u,w0) = Z(u,w0) + f(u,w0) + sum B(i,j) lambda(ij) = 0

         given the solver_kargs parameters

        '''
        
        fb = self.assemble_right_hand_side(lambda_dict)

        if u_init is None:
            u_init = self.u_init
        
        #R = lambda u : self.Z.eval(u,w0) + self.f.eval(u,w0) + fb
        R = self.residual(w0,fb)
        solution = optimize.OptimizeResult()
        
        f = R(u_init)
        if np.linalg.norm(f)<=self.solver_kargs['fatol']:
            solution.success = True
            solution.f = f
            solution.x = u_init
            self.solution = solution
            return self.solution.x

        try:
            solution = optimize.root(R,u_init,method=self.solver_kargs['method'], options=self.alg_kwargs)
        except ValueError:
            logging.error('Local Problem %i did not converge!' %self.id)        
            solution.success = False

        self.solution = solution
        # update last converged solution
        if solution.success:    
            self.u_init = self.solution.x 
            return self.u_init
        else:
            return None

    

class NonlinearSolverManager(SolverManager):
    def __init__(self,Z_dict={},B_dict={},f_dict={},local_solver={'method':'krylov','options': {}}, 
        dual_interface_algorithm={'method':'krylov','options': {}},**kwargs):
        super().__init__({},{},{},local_solver=local_solver,
                          dual_interface_algorithm=dual_interface_algorithm,**kwargs)
        
        self._create_local_problems(Z_dict,B_dict,f_dict)
        self._lambda_init = None

    def _create_local_problems(self,Z_dict,B_dict,f_dict):
        ''' Create dict of local problem, based on 
        K_dict,B_dict,f_dict dictionaries of arrays
        '''
        
        for key, Z in Z_dict.items():
            B = B_dict[key]
            f = f_dict[key]
            local_problem = NonLinearLocalProblem(Z, B, f, Z.shape[0])
            self.add_localproblem(local_problem,local_id=key)
            
    @property
    def lambda_init(self):
        return self._lambda_init
    
    @lambda_init.setter
    def lambda_init(self,lambda_init):
        self._lambda_init = lambda_init

    def solve_dual_interface_problem(self, method='newton', options=None, **kwargs ):
        ''' Strategy to solve the dual interface problem

        given lamnda_n, u_init, solve u_n implicitily based on local
        equilibrium equation

        R(u,lambda_n,w0) = 0  -> u_ solution for fixed lambda, and w0

        update lambda based on Newton-Krylov solver

        lambda_n1 = lambda_n1 + delta

        '''
        lambda_init = self.lambda_init
        #w0=np.array([0])
        R = lambda lambda_ker : self.solve_local_equilibrium(lambda_ker, **kwargs)
        
        r0 = R(lambda_init)

        # it is not required if solution converged
        #u_dict = self.get_updated_u_dict()

        self.apply_F(lambda_init,**kwargs)
        Finv = self.apply_F_inv(**kwargs)
        delta_l = Finv.solve(r0)
        
        Jac_operator = self.create_linearized_F_operator(r0,lambda_init,**kwargs)
        algorithm = getattr(opt,method)
        lambda_opt = algorithm(R,Finv,lambda_init)

        return lambda_opt

    def get_updated_u_dict(self):
        ''' get the displacament solutio for the given lambda

        '''
        u_dict = {}
        for problem_id, local_problem in self.local_problem_dict.items():    
            if local_problem.solution.success:
                u = local_problem.u_init
                u_dict.update({problem_id : u})

            else:
                logging.error('Local Problem did not converge. Do something do correct it!!')
                
        return u_dict

    def solve_local_equilibrium(self, v, u_init_dict=None,**kwargs):
        ''' apply the nonlinear F action operator to lambda F(lambda) - > delta_u
            gap = F(lambda) - d 
        
            F = sum B(i,j) d ui / d lambda ij

            Where d ui / d lambda ij is a implicit derivative
            given by the residual equilibrium equation

            R(u,w) - B.T * lambda = 0

        '''
       
        v_dict = self.vector2localdict(v, self.global2local_lambda_dofs)
        gap_dict = self.solve_interface_gap(v_dict, u_init_dict,**kwargs)
        
        d = np.zeros(self.lambda_size, dtype=self.dtype )
        for interface_id,global_index in self.local2global_lambda_dofs.items():
            d[global_index] += gap_dict[interface_id]

        return -d
        
    def solve_interface_gap(self,v_dict=None,u_init_dict=None,**kwargs):
        ''' Given a lambda dict, named v_dict solve the interface gap
        '''
        u_interface_dict = {}
        for problem_id, local_problem in self.local_problem_dict.items():    
            if u_init_dict is None:
                u_init = None
            else:
                u_init = u_init_dict[problem_id]
            u_dict_local = local_problem.solve_interface_displacement(v_dict,w0=kwargs['w0'],u_init=u_init)
            

            u_interface_dict.update(u_dict_local)

        return self.u_dict_2_gap(u_interface_dict)

    def u_dict_2_gap(self,u_dict):
        ''' Given the dict of displacement of local problems
        compute the interface gap
        '''
        # compute gap
        gap_dict = {}
        for interface_id in u_dict:
            local_id, nei_id = interface_id
            if nei_id>local_id:
                gap = u_dict[local_id,nei_id] + u_dict[nei_id,local_id]
                gap_dict[local_id, nei_id] = gap
                gap_dict[nei_id, local_id] = -gap
        return gap_dict

    def apply_F(self,v,w0):
        v_dict = self.vector2localdict(v, self.global2local_lambda_dofs)

        u_interface_dict = {}
        for problem_id, local_problem in self.local_problem_dict.items():    
            u_dict_local = local_problem.apply_local_F(v_dict,w0)
            u_interface_dict.update(u_dict_local)

        gap_dict = self.u_dict_2_gap(u_interface_dict)
        
        d = np.zeros(self.lambda_size, dtype=self.dtype )
        for interface_id,global_index in self.local2global_lambda_dofs.items():
            d[global_index] += gap_dict[interface_id]

        return -d


    def apply_F_inv(self,w0,method='cg'):

        Fdot = lambda v : self.apply_F(v,w0)
        F = sparse.linalg.LinearOperator(shape=(self.lambda_size,self.lambda_size), matvec = Fdot )

        F_inv = lambda r : sparse.linalg.cg(F,r)[0]
        return opt.LinearSolver(solver = F_inv)


    def create_linearized_F_operator(self,r,v,w0,method='FETI',options={}):
        ''' get the linearized F action based on fixed lambda
        and update it based on the residual, such that

        lambda_n1 = lambda_n + F^-1  r
        F is evaluated in lamnda_n, and w0

        return 
            F such that
            F^-1 = F.solve(r0)
        
        '''
        #r_dict = self.vector2localdict(r, self.global2local_lambda_dofs)
        #v_dict = self.vector2localdict(v, self.global2local_lambda_dofs)

        try:
            algorithm_class = getattr(opt,method)
            algorithm = algorithm_class()
            algorithm.map = self.global2local_lambda_dofs
            options.update({'w0': w0, 'v' : v})
            algorithm.options = options
            algorithm.local_problem = self.local_problem_dict
        
            
        except NotImplementedError:
            raise NotImplementedError('Option is not implemented')

        return algorithm

