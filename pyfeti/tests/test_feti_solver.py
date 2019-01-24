
import sys 
import numpy as np
from unittest import TestCase, main
from collections import OrderedDict
from scipy import sparse
sys.path.append('../..')
from pyfeti.src.utils import OrderedSet, Get_dofs, save_object
from pyfeti.src.linalg import Matrix, Vector



class  Test_FETIsolver(TestCase):
    def setUp(self):
        from pyfeti.cases.case1 import K1, K2, B1_dict, B2_dict, global_to_local_dict_1, global_to_local_dict_2, dofs_dict, map_dofs, K_global
        from pyfeti.src.feti_solver import ParallelFETIsolver, SerialFETIsolver

        # mapping global dict to local dict:
        dofs_dict_1 = OrderedDict()
        dofs_dict_2 = OrderedDict()
        for key in dofs_dict:
            global_dofs = dofs_dict[key]

            get_global_dof_row_index = lambda global_dof : list(map_dofs[map_dofs['Global_dof_id']==global_dof].index.values.astype(int))
            row2local_dof = lambda row_id : map_dofs['Local_dof_id'].ix[row_id]
            row2domain_id = lambda row_id : map_dofs['Domain_id'].ix[row_id]
            global2local_dof = lambda global_dof : (list(map(row2local_dof,get_global_dof_row_index(global_dof))), list(map(row2domain_id,get_global_dof_row_index(global_dof))))

            def get_local_dof(global_dof, domain_id):
                local_dofs_list, domain_id_list = global2local_dof(global_dof)
                try:
                    return local_dofs_list[domain_id_list.index(domain_id)]
                except:
                    return 

            get_dirichlet_local_dofs_1 = OrderedSet(list(map(lambda global_dof : get_local_dof(global_dof,1) , global_dofs)))
            get_dirichlet_local_dofs_2 = OrderedSet(list(map(lambda global_dof : get_local_dof(global_dof,2) , global_dofs)))

            dofs_dict_1[key] = get_dirichlet_local_dofs_1
            dofs_dict_2[key] = get_dirichlet_local_dofs_2


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
        f1.replace_elements('neu_y',1E3)
        f_global.replace_elements('neu_x',1E6)
        f_global.replace_elements('neu_y',1E3)


        K_dict = {}
        K_dict[1] = K1obj.data
        K_dict[2] = K2obj.data

        B_dict = {}
        B_dict[1] = B1_dict
        B_dict[2] = B2_dict

        f_dict = {}
        f_dict[1] = f1
        f_dict[2] = f2

        solver_obj  = SerialFETIsolver(K_dict,B_dict,f_dict)
        solver_obj.solve()
        

if __name__=='__main__':

    #main()
    test_obj = Test_FETIsolver()
    test_obj.setUp()