import os
from pyfeti.src.utils import save_object, load_object, pyfeti_dir
from pyfeti.src.linalg import Matrix
import copy
from scipy import sparse
import numpy as np

case_dict = {}
case_dict[1] = 'case_18'
case_dict[2] = 'case_162'
case_dict[3] = 'case_200'
case_dict[4] = 'case_800'
case_dict[5] = 'case_3200'
case_dict[6] = 'case_5000'
case_dict[7] = 'case_20000'
case_dict[8] = 'case_80000'


def get_case_matrices(case_id):
    case_path = pyfeti_dir(os.path.join('cases/matrices',case_dict[case_id]))
    K = load_object(os.path.join(case_path,'K.pkl'))
    f = load_object(os.path.join(case_path,'f.pkl'))
    B_left = load_object(os.path.join(case_path,'B_left.pkl'))
    B_right = load_object(os.path.join(case_path,'B_right.pkl'))
    B_bottom = load_object(os.path.join(case_path,'B_bottom.pkl'))
    B_top = load_object(os.path.join(case_path,'B_top.pkl'))
    s = load_object(os.path.join(case_path,'selectionOperator.pkl'))
    return K, f, B_left, B_right, B_bottom, B_top, s
    

class FETIcase_builder():
    def __init__(self,domains_x,domains_y, K, f, B_dict,s):
        self.K = K
        self.f = f
        self.B_dict = B_dict       
        self.s = s
        self.domains_x = domains_x
        self.domains_y = domains_y

    def two2one_map(self,tuple_index):
        a = 1
        b = self.domains_x
        c = 1
        i = tuple_index[0]
        j = tuple_index[1]
        if 0<=i<self.domains_x and 0<=j<self.domains_y:
            I = a*i + b*j + c
        else:
            I = None
        return I

    def get_neighbors_dict(self,I,J):
       
        neighbors_dict = {}
        neighbors_dict['right'] = (I+1,J)
        neighbors_dict['left'] = (I-1,J)
        neighbors_dict['top'] = (I,J+1)
        neighbors_dict['bottom'] = (I,J-1)
        neighbors_dict['bottom_left_corner'] = (I-1,J-1)
        neighbors_dict['bottom_right_corner'] = (I+1,J-1)
        neighbors_dict['top_right_corner'] = (I+1,J+1)
        neighbors_dict['top_left_corner'] = (I-1,J+1)

        return neighbors_dict

    def build_subdomain_matrices(self):

        K_dict = {}
        B_dict = {}
        f_dict = {}
        for j in range(self.domains_y):
            for i in range(self.domains_x):
                Neumann_mult = 0.0
                K = copy.deepcopy(self.K)
                f = copy.deepcopy(self.f)
                global_id = self.two2one_map((i,j))
                if i==0:
                    #apply dirichelt B.C
                    K_dir_obj = Matrix(K.todense(),self.s.selection_dict)
                    try:
                        K = sparse.csr_matrix(K_dir_obj.eliminate_by_identity('left'))
                    except:
                        K = sparse.csr_matrix(K_dir_obj.eliminate_by_identity(1))

                if i==(self.domains_x-1):
                    Neumann_mult = 1.0

                K_dict[global_id] = K
                f_dict[global_id] = Neumann_mult*f
                B_dict[global_id] = {}
                for bool_key, nei_index in self.get_neighbors_dict(i,j).items():
                    global_nei_id = self.two2one_map(nei_index)
                    if global_nei_id is not None:
                        try:
                            B_dict[global_id][global_id,global_nei_id] = np.sign(global_nei_id-global_id)*self.B_dict[bool_key]
                        except:
                            pass

        return K_dict, B_dict, f_dict



class CreateFETIcase(FETIcase_builder):
    def __init__(self,domains_x,domains_y, K, f, B_left, B_right, B_bottom, B_top,s):
        B_dict = {}
        B_dict['left'] = B_left
        B_dict['right'] = B_right
        B_dict['bottom'] = B_bottom
        B_dict['top'] = B_top
        super().__init__(domains_x,domains_y, K, f, B_dict ,s)

def create_FETI_case(case_id,dim_x,dim_y):
    K, f, B_left, B_right, B_bottom, B_top, s = get_case_matrices(case_id)
    print('Subdomain matrix size: [%i,%i]' %K.shape)
    case_obj = CreateFETIcase(dim_x,dim_y,K, f, B_left, B_right, B_bottom, B_top, s)
    K_dict, B_dict, f_dict = case_obj.build_subdomain_matrices()
    return K_dict, B_dict, f_dict

if __name__ == '__main__':
    case_id = 1
    dim_x,dim_y = 2,2
    #K, f, B_left, B_right, B_bottom, B_top, s = get_case_matrices(case_id)
    #case_obj = CreateFETIcase(2,1,K, f, B_left, B_right, B_bottom, B_top, s)
    #K_dict, B_dict, f_dict = case_obj.build_subdomain_matrices()
    K_dict, B_dict, f_dict = create_FETI_case(case_id,dim_x,dim_y)



