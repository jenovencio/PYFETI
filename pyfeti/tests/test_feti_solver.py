
import sys 
import numpy as np
from unittest import TestCase, main
from collections import OrderedDict
from scipy import sparse
sys.path.append('../..')
from pyfeti.src.utils import OrderedSet, Get_dofs, save_object, MapDofs
from pyfeti.src.linalg import Matrix, Vector,  elimination_matrix_from_map_dofs, expansion_matrix_from_map_dofs
from pyfeti.src.feti_solver import ParallelFETIsolver, SerialFETIsolver
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

        self.domain_list = list(self.f_dict.keys())
        self.domain_list.sort()

        
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
        np.testing.assert_almost_equal(u_global,u_primal,decimal=10)
        np.testing.assert_almost_equal(u_dual,u_dual_calc,decimal=10)

    def test_serial_solver(self):
        print('Testing Serial FETI solver ..........\n\n')
        solver_obj = SerialFETIsolver(self.K_dict,self.B_dict,self.f_dict)
        sol_obj = solver_obj.solve()
        self.postproc(sol_obj)
        print('end Serial FETI solver ..........\n\n')

    def test_elimination_matrix(self):
        solver_obj = SerialFETIsolver(self.K_dict,self.B_dict,self.f_dict)
        #manager = solver_obj.manager
        #manager.create_local_problems(solver_obj.K_dict,solver_obj.B_dict,solver_obj.f_dict)
        #B = manager.assemble_global_B_and_L()
        #L = manager.assemble_global_L()
        #K_global, f_global = manager.assemble_global_K_and_f()
    
    def test_parallel_solver(self):
        print('Testing Parallel FETI solver ..........\n\n')
        solver_obj = ParallelFETIsolver(self.K_dict,self.B_dict,self.f_dict)
        sol_obj = solver_obj.solve()
        self.postproc(sol_obj)
        print('end Parallel FETI solver ..........\n\n')

    def test_parallel_solver_cases(self):
        print('Testing Parallel FETI solver ..........\n\n')
        K_dict, B_dict, f_dict = create_FETI_case(1,4,1)
        solver_obj = ParallelFETIsolver(K_dict,B_dict,f_dict)
        sol_obj = solver_obj.solve()
        x = 1

if __name__=='__main__':

    #main()
    test_obj = Test_FETIsolver()
    test_obj.setUp()
    test_obj.test_parallel_solver_cases()
    #test_obj.test_parallel_solver()
    #test_obj.test_serial_solver()
    #test_obj.test_elimination_matrix()