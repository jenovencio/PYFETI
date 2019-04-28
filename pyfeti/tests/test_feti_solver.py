
import sys 
import numpy as np
from unittest import TestCase, main
from collections import OrderedDict
from scipy import sparse
import time
sys.path.append('../..')
from pyfeti.src.utils import OrderedSet, Get_dofs, save_object, MapDofs
from pyfeti.src.linalg import Matrix, Vector,  elimination_matrix_from_map_dofs, expansion_matrix_from_map_dofs
from pyfeti.src.feti_solver import ParallelFETIsolver, SerialFETIsolver
from pyfeti.src.solvers import PCPG
from pyfeti.cases.case_generator import create_FETI_case



K1 = np.array([[4., 0., 0., 0.],
                [0., 4., -1., -2.],
                [0., -1., 4., -1.],
                [0., -2., -1., 4.]])

K2 = K3 = K4 = np.array([[4., -1., -2., -1.],
                            [-1., 4., -1., -2.],
                            [-2., -1., 4., -1.],
                            [-1., -2., -1., 4.]])


q0 = 1.0
q1 = np.array([0.,0.,0.,0.])
q2 = np.array([0.,0.,0.,0.])
q3 = np.array([0.,0.,0.,0.])
q4 = np.array([0.,0.,1.0,0.0])*q0

B12 =  np.array([[0,1,0,0],
                    [0,0,1,0]])

B13 = np.array([[0,0,1,0],
                [0,0,0,1]])

B14 = np.array([[0,0,1,0]])

B21 =  np.array([[-1,0,0,0],
                    [0,0,0,-1]])

B23 = np.array([[0,0,0,1]])

B24 = np.array([[0,0,1,0],
                [0,0,0,1]])


B31 = np.array([[0,-1,0,0],
                [-1,0,0,0]])

B32 = np.array([[0,-1,0,0]])

B34 = np.array([[0,1,0,0],
                [0,0,1,0]])

B41 = np.array([[-1,0,0,0]])

B42 = np.array([[0,-1,0,0],
                [-1,0,0,0]])

B43 = np.array([[-1,0,0,0],
                [ 0,0,0,-1]])


class  Test_FETIsolver(TestCase):
    def setUp(self):
        from pyfeti.cases.case1 import K1, K2, B1_dict, B2_dict, global_to_local_dict_1, global_to_local_dict_2, dofs_dict, map_dofs, K_global
        

        # mapping global dict to local dict:
        dofs_dict_1 = OrderedDict()
        dofs_dict_2 = OrderedDict()
        self.map_dofs = map_dofs
        map_obj = MapDofs(map_dofs)
        for key in dofs_dict:
            global_dofs = dofs_dict[key]
            get_dirichlet_local_dofs_1 = OrderedSet(list(map(lambda global_dof : map_obj.get_local_dof(global_dof,1) , global_dofs)))
            get_dirichlet_local_dofs_2 = OrderedSet(list(map(lambda global_dof : map_obj.get_local_dof(global_dof,2) , global_dofs)))

            dofs_dict_1[key] = get_dirichlet_local_dofs_1
            dofs_dict_2[key] = get_dirichlet_local_dofs_2


        self.L = L = elimination_matrix_from_map_dofs(map_dofs)
        self.Lexp = Lexp = expansion_matrix_from_map_dofs(map_dofs)

        K1obj = Matrix(K1,dofs_dict_1)
        K1obj.eliminate_by_identity('dirichlet')

        K2obj = Matrix(K2,dofs_dict_2)
        K2obj.eliminate_by_identity('dirichlet')

        K_global_obj = Matrix(K_global.todense(),dofs_dict)
        K_global_obj.eliminate_by_identity('dirichlet')
        
        f1 = Vector(K1obj.shape[0]*[0.0],dofs_dict_1)
        f2 = Vector(K2obj.shape[0]*[0.0],dofs_dict_2)
        f_global = Vector(K_global_obj.shape[0]*[0.0],dofs_dict,name='f_global')

        f1.replace_elements('neu_x',1E6)
        f1.data[12] = 249999.99999935
        f1.data[16] = 250000.00000065
        f1.data[14] = 500000
        #f1.replace_elements('neu_y',1E3)
        #f_global.replace_elements('neu_x',1E6)
        f_global.data[2] = 249999.99999935
        f_global.data[4] = 250000.00000065
        f_global.data[14] = 500000

        self.f_global = f_global
        #f_global.replace_elements('neu_y',1E3)

        self.u_global = u_global = np.linalg.solve(K_global_obj.data,f_global.data)

        self.K_dict = {}
        self.K_dict[1] = K1obj.data
        self.K_dict[2] = K2obj.data

        self.B_dict = {}
        self.B_dict[1] = B1_dict
        self.B_dict[2] = B2_dict

        self.f_dict = {}
        self.f_dict[1] = f1
        self.f_dict[2] = f2
        
        self.domain_list = np.sort(list(self.f_dict.keys()))
        
    def test_simple_bar_problem(self):
        K1 = np.array([[2.,-1.],[-1.,1.]])
        K2 = np.array([[1.,-1.],[-1.,2.]])
        B1 = np.array([[0.,1]]) 
        B2 = np.array([[-1,0]]) 
        f1 = np.array([0.,0.])                
        f2 = np.array([0.,1.]) 

        K_dict = {1:K1,2:K2}
        B_dict = {1 : {(1,2) : B1}, 2 : {(2,1) : B2}}
        f_dict = {1:f1,2:f2}
        solver_obj = SerialFETIsolver(K_dict,B_dict,f_dict)
        solution_obj = solver_obj.solve()
        u_dual = solution_obj.displacement
        lambda_ = solution_obj.interface_lambda
        alpha =  solution_obj.alpha

        u_target = np.array([0.25,0.5,0.5,0.75])
        np.testing.assert_almost_equal(u_dual,u_target,decimal=10)

    def test_simple_bar_with_redundante_contraints(self):
        '''
        A simple example with Reduntant Constraints Positive Define Domains
        '''
        K1 = np.array([[2,-1],[-1,1]])
        K2 = np.array([[1,-1],[-1,2]])
        B1 = np.array([[0,1],[0,1],[0,1]]) 
        B2 = np.array([[-1,0],[-1,0],[-1,0]]) 
        f1 = np.array([0.,0.])                
        f2 = np.array([0.,1.])  

        K_dict = {1:K1,2:K2}
        B_dict = {1 : {(1,2) : B1}, 2 : {(2,1) : B2}}
        f_dict = {1:f1,2:f2}
        solver_obj = SerialFETIsolver(K_dict,B_dict,f_dict)
        solution_obj = solver_obj.solve()
        u_dual = solution_obj.displacement
        lambda_ = solution_obj.interface_lambda
        alpha =  solution_obj.alpha

        u_target = np.array([0.25,0.5,0.5,0.75])
        np.testing.assert_almost_equal(u_dual,u_target,decimal=10)

    def _test_2d_thermal_problem(self):

        # Using PyFETI to solve the probrem described above
        num_domain = 3
        if num_domain==4:
            K_dict = {1:K1,2:K2, 3:K3, 4:K4}
            B_dict = {1 : {(1,2) : B12, (1,3) : B13, (1,4) : B14}, 
                      2 : {(2,1) : B21, (2,4) : B24,(2,3) : B23}, 
                      3 : {(3,1) : B31, (3,4) : B34, (3,2) : B32}, 
                      4 : {(4,2) : B42, (4,3) : B43, (4,1) : B41}}

            q_dict = {1:q1 ,2:q2, 3:q3, 4:q4}

        elif num_domain==2:
            K_dict = {1:K1,2:K2}
            B_dict = {1 : {(1,2) : B12}, 
                      2 : {(2,1) : B21}}

            q_dict = {1:q1 ,2:q4}

        elif num_domain==3:
            K_dict = {1:K1,2:K2,3:K3}
            B_dict = {1 : {(1,2) : B12}, 
                      2 : {(2,1) : B21, (2,3) : B12},
                      3 : {(3,2) : B21 }}
            
            q_dict = {1:q1 ,2:q2, 3:q4}

        solver_obj = SerialFETIsolver(K_dict,B_dict,q_dict)

        L = solver_obj.manager.assemble_global_L()
        Lexp = solver_obj.manager.assemble_global_L_exp()
        B = solver_obj.manager.assemble_global_B()
        K, f = solver_obj.manager.assemble_global_K_and_f()
        R = solver_obj.manager.assemble_global_kernel()
        e = solver_obj.manager.assemble_e()
        G = solver_obj.manager.assemble_G()
        GGT_inv = np.linalg.inv(G.dot(G.T))
        P = np.eye(B.shape[0]) - (G.T.dot(GGT_inv)).dot(G)
        F_feti = solver_obj.manager.assemble_global_F()

        f_primal = L.dot(f)
        K_primal = L.dot(K.dot(Lexp))
        T_primal = np.linalg.solve(K_primal,f_primal)

        T_dual = Lexp.dot(T_primal)
        interface_gap = B.dot(T_dual)
        np.testing.assert_almost_equal(interface_gap,0*interface_gap,decimal=10)

        K_inv = np.linalg.pinv(K.A)

        F = B@K_inv@B.T
        d = B@K_inv@f

        lambda_im = G.T.dot(GGT_inv).dot(e)
        r0 = d - F.dot(lambda_im)
        Fp = P.T.dot(F.dot(P))
        dp = P.T.dot(d)
        lambda_ker, info = sparse.linalg.cg(Fp,dp,M=P)
        #lambda_ker, info = sparse.linalg.minres(Fp,dp,M=P)
        F_action = lambda x : F.dot(x)
        Projection_action = lambda x : P.dot(x)
        lampda_ker, rk, proj_r_hist, lambda_hist = PCPG(F_action,r0,Projection_action,tolerance=1.e-16,max_int=1000)

        lambda_cg = lambda_im + lambda_ker
        r = d - F.dot(lambda_cg)
        alpha = GGT_inv.dot(G.dot(r))

        T_cg = K_inv@(f - B.T@lambda_cg) + R.dot(alpha)

        np.testing.assert_almost_equal(T_cg,T_dual,decimal=10)

    def test_verify_F_operator(self):
        K_dict = {1:K1,2:K2, 3:K3, 4:K4}
        B_dict = {1 : {(1,2) : B12, (1,3) : B13, (1,4) : B14}, 
                      2 : {(2,1) : B21, (2,4) : B24,(2,3) : B23}, 
                      3 : {(3,1) : B31, (3,4) : B34, (3,2) : B32}, 
                      4 : {(4,2) : B42, (4,3) : B43, (4,1) : B41}}

        q_dict = {1:q1 ,2:q2, 3:q3, 4:q4}

        solver_obj = SerialFETIsolver(K_dict,B_dict,q_dict)

        L = solver_obj.manager.assemble_global_L()
        Lexp = solver_obj.manager.assemble_global_L_exp()
        B = solver_obj.manager.assemble_global_B()
        K, f = solver_obj.manager.assemble_global_K_and_f()
        R = solver_obj.manager.assemble_global_kernel()
        e = solver_obj.manager.assemble_e()
        G = solver_obj.manager.assemble_G()
        GGT_inv = np.linalg.inv(G.dot(G.T))
        P = np.eye(B.shape[0]) - (G.T.dot(GGT_inv)).dot(G)
        F_feti = solver_obj.manager.assemble_global_F()

        K_inv = np.linalg.pinv(K.A)
        F = B@K_inv@B.T
        d = B@K_inv@f

        x = np.ones(F.shape[0])
        np.testing.assert_almost_equal(F.dot(x),F_feti.dot(x),decimal=10)

    def dual2primal(self,K_dual,u_dual,f_dual,L,Lexp):
        
        f_primal = L@f_dual
        K_primal = L@K_dual@Lexp
        u_primal = np.linalg.solve(K_primal,f_primal)
        u_dual_calc = Lexp@u_primal
        norm1 = np.linalg.norm(u_dual)
        norm2 = np.linalg.norm(u_dual_calc)
        np.testing.assert_almost_equal(u_dual/norm2,u_dual_calc/norm2,decimal=8)
        return u_primal

    def postproc(self,sol_obj):

        domain_list = self.domain_list
        L = self.L
        Lexp = self.Lexp
    
        u_dict  = sol_obj.u_dict
        lambda_dict = sol_obj.lambda_dict
        alpha_dict = sol_obj.alpha_dict
        
        

        u_dual = sol_obj.displacement
        u_global = self.u_global
        u_primal = L.dot(u_dual)
        u_dual_calc = Lexp.dot(u_global)

        interface_gap = self.B_dict[1][1,2]*u_dict[1] + self.B_dict[2][2,1]*u_dict[2]
        np.testing.assert_almost_equal(interface_gap,0*interface_gap,decimal=10)
        self.check_interface_gap(u_dict,self.B_dict)

        np.testing.assert_almost_equal(u_global,u_primal,decimal=10)
        np.testing.assert_almost_equal(u_dual,u_dual_calc,decimal=10)

    def check_interface_gap(self,u_dict,B_dict):
        for key, B in  B_dict.items():
            for (domain_id,nei_id) in B:
                if nei_id > domain_id:
                    interface_gap = B_dict[domain_id][domain_id,nei_id]*u_dict[domain_id] \
                        + B_dict[nei_id][nei_id,domain_id]*u_dict[nei_id]
                    np.testing.assert_almost_equal(interface_gap,0*interface_gap,decimal=10)

    def test_serial_preconditioner(self):

        u_dual, sol_obj_1 = self.test_serial_solver(precond_type=None)
        u_dual_lumped, sol_obj_2 = self.test_serial_solver(precond_type='Lumped')
        u_dual_dir, sol_obj_3 = self.test_serial_solver(precond_type='Dirichlet')

        np.testing.assert_almost_equal( u_dual, u_dual_lumped,decimal=10)
        np.testing.assert_almost_equal( u_dual, u_dual_dir,decimal=10)

        self.assertTrue(sol_obj_1.PCGP_iterations>=sol_obj_2.PCGP_iterations)
        self.assertTrue(sol_obj_2.PCGP_iterations>=sol_obj_3.PCGP_iterations)

    def test_serial_solver(self,precond_type=None):
        print('Testing Serial FETI solver ..........\n\n')
        solver_obj = SerialFETIsolver(self.K_dict,self.B_dict,self.f_dict,precond_type=precond_type,tolerance=1E-11)
        sol_obj = solver_obj.solve()
        self.postproc(sol_obj)

        K_dual = sparse.block_diag((self.K_dict[1],self.K_dict[2]))
        f_dual = np.concatenate((self.f_dict[1].data,self.f_dict[2].data))
        u_dual = sol_obj.displacement

        np.testing.assert_almost_equal(self.f_global.data,self.L@f_dual,decimal=10)
        u_primal = self.dual2primal(K_dual,u_dual,f_dual,self.L,self.Lexp)
        np.testing.assert_almost_equal( self.u_global,u_primal,decimal=10)

        print('end Serial FETI solver ..........\n\n')
        return u_dual, sol_obj


    def test_elimination_matrix(self):

        self.setUp()
        L_target = self.L
        Lexp_target = self.Lexp  
        solver_obj = SerialFETIsolver(self.K_dict,self.B_dict,self.f_dict)
        sol_obj = solver_obj.solve()
        
        u_dual = np.array([])
        u_dict  = sol_obj.u_dict
        lambda_dict = sol_obj.lambda_dict
        alpha_dict = sol_obj.alpha_dict
        
        for domain_id in self.domain_list:
            u_dual = np.append(u_dual,u_dict[domain_id])


        L_calc = solver_obj.manager.assemble_global_L()
        Lexp_calc = solver_obj.manager.assemble_global_L_exp()

        n = L_target .shape[0]
        #testing orthogonality
        np.testing.assert_array_almost_equal(L_calc.dot(Lexp_calc),np.eye(n))

        # testing L matrix asseblying method
        np.testing.assert_array_almost_equal(np.sort(L_calc.dot(u_dual)),np.sort(self.u_global))
    
    def test_parallel_solver(self):
        print('Testing Parallel FETI solver ..........\n\n')
        solver_obj = ParallelFETIsolver(self.K_dict,self.B_dict,self.f_dict)
        sol_obj = solver_obj.solve()
        self.postproc(sol_obj)
        print('end Parallel FETI solver ..........\n\n')

    def test_parallel_solver_cases(self):
        solver_obj,sol_obj = self.run_solver_cases(algorithm=ParallelFETIsolver)
        try:
            solver_obj.manager.delete()
        except:
            print('Could not delete parallel temp folder!')

    def test_serial_solver_cases(self):
        self.run_solver_cases(algorithm=SerialFETIsolver)

    def test_serial_solver_cases_precond(self):
        solver_obj,sol_obj_1 = self.run_solver_cases(algorithm=SerialFETIsolver,precond_type=None)
        solver_obj,sol_obj_2 = self.run_solver_cases(algorithm=SerialFETIsolver,precond_type="Lumped")
        solver_obj,sol_obj_3 = self.run_solver_cases(algorithm=SerialFETIsolver,precond_type="Dirichlet")
        solver_obj,sol_obj_4 = self.run_solver_cases(algorithm=SerialFETIsolver,precond_type="LumpedDirichlet")

        np.testing.assert_almost_equal( sol_obj_1.displacement, sol_obj_2.displacement,decimal=10)
        np.testing.assert_almost_equal( sol_obj_1.displacement, sol_obj_3.displacement,decimal=10)

        self.assertTrue(sol_obj_1.PCGP_iterations>=sol_obj_2.PCGP_iterations)
        self.assertTrue(sol_obj_2.PCGP_iterations>=sol_obj_3.PCGP_iterations)

    def run_solver_cases(self,algorithm=SerialFETIsolver,precond_type=None):

        domin_list_x = [4] 
        domin_list_y = [4] 
        case_id_list = [1,2] 
        for case_id in case_id_list:
            for ny in domin_list_y:
                for nx in domin_list_x:
                    print('Testing %s ..........' %algorithm.__name__)
                    print('Number of Subdomain in X direction : %i' %nx)
                    print('Number of Subdomain in Y direction : %i ' %ny)
                    K_dict, B_dict, f_dict = create_FETI_case(case_id,nx,ny)
                    
                    solver_obj = algorithm(K_dict,B_dict,f_dict,dual_interface_algorithm='PCPG',precond_type=precond_type)
                    start_time = time.time()
                    sol_obj = solver_obj.solve()
                    elapsed_time = time.time() - start_time
                    print('Elapsed time : %f ' %elapsed_time)
                    u_dual,lambda_,alpha = self.postprocessing(sol_obj,solver_obj)
                    print('end %s ..........\n\n\n' %algorithm.__name__)

        return solver_obj,sol_obj
      
    def test_compare_serial_and_parallel_solver_slusps(self):
        pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-8}
        self.test_compare_serial_and_parallel_solver(pseudoinverse_kargs=pseudoinverse_kargs)

    def test_compare_svd_splusps(self):
        print('Starting Comparison between Serial SVD and SPLUSPS ..........')
        case_id,nx,ny = 4,4,4
        print('Critical Case Selected %i ' %case_id)
        print('Number of Domain in the X-direction %i ' %nx)
        print('Number of Domain in the Y-direction %i ' %ny)
        K_dict, B_dict, f_dict = create_FETI_case(case_id,nx,ny)
        
        solver_obj = SerialFETIsolver(K_dict,B_dict,f_dict,pseudoinverse_kargs={'method':'svd','tolerance':1.0E-8})
        start_time = time.time()
        print('....................................')
        print('Starting SVD FETI solver ..........')
        sol_obj = solver_obj.solve()
        elapsed_time = time.time() - start_time
        print('SVD Solver : Elapsed time : %f ' %elapsed_time)
        u_dual_svd,lambda_svd,alpha_svd = self.obj_to_array(sol_obj)


        print('\n\n Starting SPLUSPS FETI solver ..........')
        solver_obj = SerialFETIsolver(K_dict,B_dict,f_dict,pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-8})
        start_time = time.time()
        sol_obj_slu = solver_obj.solve()
        elapsed_time = time.time() - start_time
        print('SPLUSPS Solver : Elapsed time : %f ' %elapsed_time)
        print('....................................')

        # check gap using SPLUSPS local solver
        self.check_interface_gap(sol_obj_slu.u_dict,solver_obj.B_dict)

        # assembling dual vectors 
        u_dual_slu,lambda_slu,alpha_slu = self.obj_to_array(sol_obj_slu)
 
        # compare results  
        norm = np.linalg.norm(u_dual_svd)
        norm_lambda = np.linalg.norm(lambda_svd)
        norm_alpha = np.linalg.norm(alpha_svd)
        np.testing.assert_almost_equal(u_dual_svd/norm,u_dual_slu/norm,decimal=10)
        np.testing.assert_almost_equal(lambda_svd/norm_lambda,lambda_slu/norm_lambda,decimal=10)
        
        print('End Comparison SVD and SPLUSPS FETI solver ..........\n\n')

    def test_compare_serial_and_parallel_solver(self,pseudoinverse_kargs={'method':'svd','tolerance':1.0E-8}):

        print('Starting Comparison between Serial and Parallel FETI solver ..........')
        case_id,nx,ny = 1,2,1
        print('Critial Case Selected %i ' %case_id)
        print('Number of Domain in the X-direction %i ' %nx)
        print('Number of Domain in the Y-direction %i ' %ny)
        K_dict, B_dict, f_dict = create_FETI_case(case_id,nx,ny)
        
        solver_obj = SerialFETIsolver(K_dict,B_dict,f_dict,pseudoinverse_kargs=pseudoinverse_kargs)
        start_time = time.time()
        print('Starting Serial FETI solver ..........')
        sol_obj = solver_obj.solve()
        elapsed_time = time.time() - start_time
        print('Serial Solver : Elapsed time : %f ' %elapsed_time)
        u_dual_serial,lambda_serial,alpha_serial = self.obj_to_array(sol_obj)

        print('\n\n Starting Parallel FETI solver ..........')
        solver_obj = ParallelFETIsolver(K_dict,B_dict,f_dict,pseudoinverse_kargs=pseudoinverse_kargs)
        start_time = time.time()
        sol_obj = solver_obj.solve()
        elapsed_time = time.time() - start_time
        print('Parallel Solver : Elapsed time : %f ' %elapsed_time)

        u_dual_parallel,lambda_parallel,alpha_parallel = self.obj_to_array(sol_obj)

        
        np.testing.assert_almost_equal(u_dual_serial,u_dual_parallel,decimal=10)
        np.testing.assert_almost_equal(lambda_serial/np.linalg.norm(lambda_serial),
                                       lambda_parallel/np.linalg.norm(lambda_parallel),decimal=10)
        np.testing.assert_almost_equal(alpha_serial,alpha_parallel,decimal=10)

        print('End Comparison Serial and Parallel FETI solver ..........\n\n')

    def obj_to_array(self,sol_obj):
        u_dict  = sol_obj.u_dict
        lambda_dict = sol_obj.lambda_dict
        alpha_dict = sol_obj.alpha_dict
        
        u_dual = sol_obj.displacement
        lambda_ = sol_obj.interface_lambda
        alpha = sol_obj.alpha
        return u_dual,lambda_,alpha

    def postprocessing(self,sol_obj,solver_obj):
        u_dict  = sol_obj.u_dict
        lambda_dict = sol_obj.lambda_dict
        alpha_dict = sol_obj.alpha_dict
        
        u_dual = sol_obj.displacement
        lambda_ = sol_obj.interface_lambda
        alpha = sol_obj.alpha

        #update L matrices
        L = solver_obj.manager.assemble_global_L()
        Lexp = solver_obj.manager.assemble_global_L_exp()

        # get dual matrices
        K_dual, f_dual = solver_obj.manager.assemble_global_K_and_f()
        
        if False:
            # get F operator
            F = solver_obj.manager.assemble_global_F()
            G = solver_obj.manager.assemble_G()
            e = solver_obj.manager.assemble_e()
            d = solver_obj.manager.assemble_global_d()

            B = solver_obj.manager.assemble_global_B()

            GGT_inv_ = np.linalg.inv(G@G.T)
            GGT_inv =  solver_obj.manager.compute_GGT_inverse() 

            lambda_im = G.T.dot((GGT_inv).dot(e))
            I = np.eye(len(lambda_im))
            P = lambda r : (I - G.T.dot(GGT_inv.dot(G))).dot(r)
            F_action = lambda x : F.dot(x)
            lambda_ker = lambda_ - lambda_im
            residual = d - F_action(lambda_im)
            lampda_pcpg, rk, proj_r_hist, lambda_hist = PCPG(F_action,residual,Projection_action=P,tolerance=1.e-10,max_int=500)
            lambda_calc = lampda_pcpg + lambda_im
            alpha_sol = GGT_inv.dot(G.dot(d - F.dot(lampda_pcpg)))


        # check error 
        B_dict = solver_obj.B_dict
        self.check_interface_gap(u_dict,B_dict)
        u_primal = self.dual2primal(K_dual,u_dual,f_dual,L,Lexp)
        
        return u_dual,lambda_,alpha

    def test_total_FETI_approach(self):
        ''' This test incorporate Dirichlet constraint in the Bollean matrix
        The constraint are considered the 0-th Neighbor
                                           F->
        |>0   0-----0-----0    0-----0-----0
        Dir        D1                D2 
        '''
        

        K1 = np.array([[1,-1],[-1,1]])
        K2 = np.array([[1,-1],[-1,1]])
        B0 = np.array([[-1,0]])
        B1 = np.array([[0,1]]) 
        B2 = np.array([[-1,0]]) 

        f1 = np.array([0.,0.])                
        f2 = np.array([0.,1.])                
               
        # Using PyFETI to solve the probrem described above
        K_dict = {1:K1,2:K2}
        B_dict = {1 : {(1,2) : B1, (1,1): B0}, 2 : {(2,1) : B2}}
        f_dict = {1:f1,2:f2}

        solver_obj = SerialFETIsolver(K_dict,B_dict,f_dict)

        solution_obj = solver_obj.solve()

        u_dual = solution_obj.displacement
        lambda_ = solution_obj.interface_lambda
        alpha =  solution_obj.alpha
        

        solver_obj = ParallelFETIsolver(K_dict,B_dict,f_dict)

        solution_obj = solver_obj.solve()

        solver_obj.manager.delete()

        u_dual_par = solution_obj.displacement
        lambda_par = solution_obj.interface_lambda
        alpha_par =  solution_obj.alpha

        np.testing.assert_almost_equal(u_dual,u_dual_par,decimal=10)
        np.testing.assert_almost_equal(lambda_,lambda_par,decimal=10)
        np.testing.assert_almost_equal(alpha,alpha_par,decimal=10)
        
        

if __name__=='__main__':

    #main()
    test_obj = Test_FETIsolver()
    #test_obj.setUp()
    #test_obj.test_serial_solver()
    #test_obj.test_serial_preconditioner()
    test_obj.test_serial_solver_cases_precond()
    #test_obj.test_parallel_solver()
    #test_obj.test_parallel_solver_cases()
    #test_obj.test_serial_solver_cases()
    #test_obj.test_elimination_matrix()
    #test_obj.test_compare_serial_and_parallel_solver()
    #test_obj.test_simple_bar_problem()
    #test_obj.test_simple_bar_with_redundante_contraints()
    #test_obj._test_2d_thermal_problem()
    #test_obj.test_verify_F_operator()
    #test_obj.test_compare_serial_and_parallel_solver_slusps()
    #test_obj.test_compare_svd_splusps()
    #test_obj.test_total_FETI_approach()