
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
        
    def dual2primal(self,K_dual,u_dual,f_dual,L,Lexp):
        
        f_primal = L@f_dual
        K_primal = L@K_dual@Lexp
        u_primal = np.linalg.solve(K_primal,f_primal)
        np.testing.assert_almost_equal(u_dual,Lexp@u_primal,decimal=10)
        return u_primal

    def postproc(self,sol_obj):

        domain_list = self.domain_list
        L = self.L
        Lexp = self.Lexp
        u_dual = np.array([])
        u_dict  = sol_obj.u_dict
        lambda_dict = sol_obj.lambda_dict
        alpha_dict = sol_obj.alpha_dict
        
        for domain_id in domain_list:
            u_dual = np.append(u_dual,u_dict[domain_id])

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

    def test_serial_solver(self):
        print('Testing Serial FETI solver ..........\n\n')
        solver_obj = SerialFETIsolver(self.K_dict,self.B_dict,self.f_dict)
        sol_obj = solver_obj.solve()
        self.postproc(sol_obj)

        K_dual = sparse.block_diag((self.K_dict[1],self.K_dict[2]))
        f_dual = np.concatenate((self.f_dict[1].data,self.f_dict[2].data))
        u_dual = sol_obj.displacement

        np.testing.assert_almost_equal(self.f_global.data,self.L@f_dual,decimal=10)
        u_primal = self.dual2primal(K_dual,u_dual,f_dual,self.L,self.Lexp)
        np.testing.assert_almost_equal( self.u_global,u_primal,decimal=10)

        print('end Serial FETI solver ..........\n\n')

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
        self.test_solver_cases(algorithm=ParallelFETIsolver)

    def test_serial_solver_cases(self):
        #print('Testing Serial FETI solver ..........\n\n')
        #K_dict, B_dict, f_dict = create_FETI_case(1,4,1)
        #solver_obj = SerialFETIsolver(K_dict,B_dict,f_dict)
        #sol_obj = solver_obj.solve()
        #u_dual,lambda_,alpha = self.postprocessing(sol_obj,solver_obj)
        #print('end Serial FETI solver ..........\n\n')
        self.test_solver_cases(algorithm=SerialFETIsolver)


    def test_solver_cases(self,algorithm=SerialFETIsolver):

        domin_list_x = [1,2,3,4,5,10]
        domin_list_y = [1]
        case_id_list =[1,2]
        for case_id in case_id_list:
            for ny in domin_list_y:
                for nx in domin_list_x:
                    print('Testing %s ..........' %algorithm.__name__)
                    print('Number of Subdomain in X direction : %i' %nx)
                    print('Number of Subdomain in Y direction : %i ' %ny)
                    K_dict, B_dict, f_dict = create_FETI_case(case_id,nx,ny)
                    
                    solver_obj = algorithm(K_dict,B_dict,f_dict)
                    start_time = time.time()
                    sol_obj = solver_obj.solve()
                    elapsed_time = time.time() - start_time
                    print('Elapsed time : %f ' %elapsed_time)
                    u_dual,lambda_,alpha = self.postprocessing(sol_obj,solver_obj)
                    print('end %s ..........\n\n\n' %algorithm.__name__)
      



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
        
        try:
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
        except:
            pass

        # check error 
        B_dict = solver_obj.B_dict
        self.check_interface_gap(u_dict,B_dict)
        u_primal = self.dual2primal(K_dual,u_dual,f_dual,L,Lexp)
        
        return u_dual,lambda_,alpha

        
        

if __name__=='__main__':

    #main()
    test_obj = Test_FETIsolver()
    test_obj.setUp()
    #test_obj.test_serial_solver()
    #test_obj.test_parallel_solver()
    #test_obj.test_parallel_solver_cases()
    test_obj.test_serial_solver_cases()
    #test_obj.test_elimination_matrix()