import sys
import numpy as np
sys.path.append('../..')
from pyfeti.src.utils import OrderedSet, Get_dofs, save_object
from pyfeti.src.linalg import Matrix, Vector
from pyfeti.src import solvers
import logging

class FETIsolver():
    def __init__(self,K_dict,B_dict,f_dict):
        self.K_dict = K_dict
        self.B_dict = B_dict
        self.f_dict = f_dict
        self.x_dict = None
        self.lambda_dict = None
        self.alpha_dict = None
        
        
    def solve(self):
        pass
        
    def serialize(self):
        for obj in [self.K_dict,self.B_dict,self.f_dict]:
            for key, item in obj:
                pass
                
class SerialFETIsolver(FETIsolver):
    def __init__(self,K_dict,B_dict,f_dict):
        super().__init__(K_dict,B_dict,f_dict)

    def solve(self):
       manager = SerialSolverManager() 
       manager.create_local_problems(self.K_dict,self.B_dict,self.f_dict)
       manager.assemble_local_G_GGT_and_e()
       manager.assemble_cross_GGT()
       manager.build_local_to_global_mapping()
       #manager.compute_local_GGT_inv()
       GGT = manager.assemble_GGT()
       G = manager.assemble_G()
       e = manager.assemble_e()
       #lambda_im = manager.compute_lambda_im()
       #gap_error = manager.apply_F(lambda_im, external_force=True)

       lambda_sol,alpha_sol, rk, proj_r_hist, lambda_hist = manager.solve_dual_interface_problem()
       u_dict, lambda_dict, alpha_dict = manager.assemble_solution_dict(lambda_sol,alpha_sol)
       return Solution(u_dict, lambda_dict, alpha_dict,rk, proj_r_hist, lambda_hist)
        
class SerialSolverManager():
    def __init__(self):
        self.local_problem_dict = {}
        self.course_probrem = CourseProblem()
        self.local2global_lambda_dofs = {}
        self.global2local_lambda_dofs = {}
        self.local2global_alpha_dofs = {}
        self.global2local_alpha_dofs = {}
        self.local_lambda_length_dict = {}
        self.local_alpha_length_dict = {}
        self.local_problem_id_list = []
        self.lambda_size = None
        self.alpha_size = None
        self.e_dict = {}
        self.G = None
        self.e = None
        self.GGT = None
        self.GGT_inv = None

    def create_local_problems(self,K_dict,B_dict,f_dict):
        for key, obj in K_dict.items():
            B_local_dict = B_dict[key]
            self.local_problem_id_list.append(key)
            self.local_problem_dict[key] = LocalProblem(obj,B_local_dict,f_dict[key],id=key)
            for interface_id, B in B_local_dict.items():
                self.local_lambda_length_dict[interface_id] = B.shape[0]
        
        self.local_problem_id_list.sort()

    def assemble_local_G_GGT_and_e(self):
        for problem_id, local_problem in self.local_problem_dict.items():
            R = local_problem.get_kernel()
            if R.shape[0]>0:
                self.e_dict[problem_id] = -R.T.dot(local_problem.f_local.data)
                self.course_probrem.update_e_dict(self.e_dict)
                self.local_alpha_length_dict[problem_id] = R.shape[1]
                G_local_dict = {}
                GGT_local_dict = {}
                for key, B_local in local_problem.B_local.items():
                    local_id, nei_id = key
                    G = (nei_id - local_id)*(-B_local.dot(R)).T
                    G_local_dict[key] = G
                    GGT_local_dict[local_id,local_id] = G.dot(G.T)
                    self.course_probrem.update_G_dict(G_local_dict)
                    self.course_probrem.update_GGT_dict(GGT_local_dict)
        
    def assemble_cross_GGT(self):
        for problem_id, local_problem in self.local_problem_dict.items():
            GGT_local_dict = {}
            for key, Gi in self.course_probrem.G_dict.items():
                local_id, nei_id = key
                for nei_id in local_problem.neighbors_id:
                    try:
                        Gj = self.course_probrem.G_dict[nei_id,local_id]
                        if Gi.shape[0]>0 and Gj.shape[0]>0:
                            GGT_local_dict[local_id,nei_id_id] = Gi.dot(Gj.T)
                            self.course_probrem.update_GGT_dict(GGT_local_dict)
                    except:
                        pass
             
    def assemble_e(self):
        try:
            self.e = self.course_probrem.assemble_e(self.local2global_alpha_dofs,self.alpha_size)
            return  
        except:
            raise('Build local to global mapping before calling this function')

    def assemble_GGT(self):
        try:
            self.GGT = self.course_probrem.assemble_GGT(self.local2global_alpha_dofs,(self.alpha_size ,self.alpha_size))
            return self.GGT
        except:
            raise('Build local to global mapping before calling this function')
    
    def assemble_G(self):
        try:
            self.G = self.course_probrem.assemble_G(self.local2global_alpha_dofs,self.local2global_lambda_dofs,(self.alpha_size ,self.lambda_size))
            return self.G
        except:
            raise('Build local to global mapping before calling this function')

    def compute_GGT_inverse(self):
        self.GGT_inv = np.linalg.inv(self.GGT)
        return self.GGT_inv
    
    def compute_lambda_im(self):
        GGT_inv = self.compute_GGT_inverse()
        return  self.G.T.dot((GGT_inv).dot(self.e))
        
    def solve_interface_gap(self,v_dict=None, external_force=False):
        u_dict = {}
        for problem_id, local_problem in self.local_problem_dict.items():
            u = local_problem.solve(v_dict,external_force)
            u_dict_local = local_problem.get_interface_dict(u)
            u_dict.update(u_dict_local)

        # compute gap
        gap_dict = {}
        for interface_id in u_dict:
            local_id, nei_id = interface_id
            if nei_id>local_id:
                gap = u_dict[local_id,nei_id] - u_dict[nei_id,local_id]
                gap_dict[local_id, nei_id] = gap
                gap_dict[nei_id, local_id] = -gap
        return gap_dict

    def compute_local_GGT_inv(self):
        self.course_probrem.compute_local_GGT_inv()

    def solve_dual_interface_problem(self,algorithm='PCPG'):
        
        lambda_im = self.compute_lambda_im()
        G = self.G
        GGT_inv = self.GGT_inv
        I = np.eye(self.lambda_size)
        Projection_action = lambda r : (I - G.T.dot(GGT_inv.dot(G))).dot(r)
        F_action = lambda lambda_ker : self.apply_F(lambda_ker)
        residual = -self.apply_F(lambda_im, external_force=True)

        method_to_call = getattr(solvers, algorithm)

        lambda_ker, rk, proj_r_hist, lambda_hist = method_to_call(F_action,residual,Projection_action=Projection_action,
                                                         lambda_init=None,
                                                         Precondicioner_action=None,
                                                         tolerance=1.e-10,max_int=500)

        lambda_sol = lambda_im + lambda_ker

        alpha_sol = GGT_inv.dot(G.dot(residual - self.apply_F(lambda_ker)))

        return lambda_sol,alpha_sol, rk, proj_r_hist, lambda_hist

    def vector2localdict(self,v,map_dict):
        v_dict = {}
        for global_index, local_info in map_dict.items():
            for interface_id in local_info:
                v_dict[interface_id] = v[np.ix_(global_index)]

        return v_dict

    def apply_F(self, v,  external_force=False):
       
        v_dict = self.vector2localdict(v, self.global2local_lambda_dofs)
        gap_dict = self.solve_interface_gap(v_dict,external_force)
        
        d = np.zeros(self.lambda_size)
        for interface_id,global_index in self.local2global_lambda_dofs.items():
            d[global_index] += gap_dict[interface_id]

        return -d

    def build_local_to_global_mapping(self):

        dof_lambda_init = 0
        dof_alpha_init = 0
        for local_id in self.local_problem_id_list:
            if local_id in self.local_alpha_length_dict:
                local_alpha_length = self.local_alpha_length_dict[local_id]
                local_alpha_dofs = np.arange(local_alpha_length) 
                global_alpha_index = dof_alpha_init + local_alpha_dofs
                dof_alpha_init+=local_alpha_length
                self.local2global_alpha_dofs[local_id] = global_alpha_index
                self.global2local_alpha_dofs[tuple(global_alpha_index)] = {local_id:local_alpha_dofs}
                #for dof in local_alpha_dofs:
                    #tuple_key = local_id,dof
                    #global_key = global_alpha_index[dof]
                    #self.global2local_alpha_dofs[global_key] = tuple_key
                    #self.local2global_alpha_dofs[tuple_key] = global_key
                    
            for nei_id in self.local_problem_dict[local_id].neighbors_id:
                if nei_id>local_id:
                    try:
                        local_lambda_length = self.local_lambda_length_dict[local_id,nei_id]
                        local_dofs = np.arange(local_lambda_length) 
                        global_index = dof_lambda_init + local_dofs
                        dof_lambda_init+=local_lambda_length
                        self.local2global_lambda_dofs[local_id,nei_id] = global_index 
                        self.global2local_lambda_dofs[tuple(global_index)] = {(local_id,nei_id):local_dofs}
                        #for dof in local_dofs:
                        #    tuple_key = local_id,nei_id,dof
                        #    global_key = global_index[dof]
                        #    #self.local2global_lambda_dofs[tuple_key] = global_key
                        #    self.global2local_lambda_dofs[global_key] = tuple_key
                    except:
                        continue
        self.lambda_size = dof_lambda_init
        self.alpha_size = dof_alpha_init

    def assemble_solution_dict(self,lambda_sol,alpha_sol):
        ''' This function creates a solution dict for interface
        primal dofs (e.g displacement) and dual variables (e.g interface forces)
        which is compatible to the problem formulation 

        K_dict = [K1,K2, ...., Kn]
        
        B_dict = [B1,B2, ....., Bn]

        where B1 is a dict, where key are interface ids

        f_dict = [f1,f2,...,fn]

        '''

        alpha_dict = {}
        lambda_dict = {}
        u_dict = {}
        for problem_id, local_problem in self.local_problem_dict.items():

            lambda_dict[problem_id] = self.vector2localdict(lambda_sol, self.global2local_lambda_dofs)
            u_local = local_problem.solve(lambda_dict[problem_id],external_force_bool=True)

            if local_problem.kernel.shape[0]>0:
                alpha_dict.update(self.vector2localdict(alpha_sol, self.global2local_alpha_dofs))
                delta_u = local_problem.rigid_body_correction(alpha_dict[problem_id])
                u_local +=delta_u

            u_dict[problem_id] = u_local

        return u_dict, lambda_dict, alpha_dict 

class ParallelFETIsolver(FETIsolver):
    def __init__(self):
        super().__init__(K_dict,B_dict,f_dict)
        
    def solve(self):
        self.serialize()
        pass
        
class LocalProblem():
    counter = 0
    def __init__(self,K_local, B_local, f_local,id):
        LocalProblem.counter+=1
        if isinstance(K_local,Matrix):
            self.K_local = K_local
        else:
            self.K_local = Matrix(K_local)

        if isinstance(f_local,Vector):
            self.f_local = f_local
        else:
            self.f_local = Vector(f_local)

        self.B_local = B_local
        self.solution = None
        
        self.id = id

        self.neighbors_id = []
        self.get_neighbors_id()

    def get_neighbors_id(self):
        for nei_id, obj in self.B_local.items():
            self.neighbors_id.append(nei_id[1])
        self.neighbors_id.sort()

    def solve(self, lambda_dict,external_force_bool=False):
       
        if not external_force_bool:
            f = np.zeros(self.f_local.data.shape)
        else:
            f = np.copy(self.f_local.data)

        if lambda_dict is not None:
            # assemble interface and external forces
            for interface_id, f_interface in lambda_dict.items():
                (local_id,nei_id) = interface_id
                if local_id!=self.id:
                    interface_id = (nei_id,local_id) 
                f -= self.B_local[interface_id].T.dot(f_interface)
        return self.K_local.apply_inverse(f)

    def get_interface_dict(self,x):
        interface_dict = {}
        for interface_id, Bij in self.B_local.items():
            (local_id,nei_id) = interface_id
            #interface_dict[interface_id] = (nei_id-local_id)*Bij.dot(x)
            interface_dict[interface_id] = (nei_id-local_id)*Bij.dot(x)

        return interface_dict

    def rigid_body_correction(self,alpha):
        return self.get_kernel().dot(alpha)
        
    def primal_interface_solution(self):
        pass
        
    @property
    def kernel(self):
        return self.get_kernel()

    def get_kernel(self):
        ''' get the rigid body bases (kernel) of locals
        problem.
        
        return Matrix 
            kernel bases of K_local
        '''
        return self.K_local.kernel
    
    def get_interface_kernel(self):
        ''' Project the Rigid body modes (R)
        in the interface dofs (G = (RG)^T)
        '''
        pass
        
    def compute_null_space_force(self):
        ''' compute null space force (e = R'f)
        '''
        pass
        
class CourseProblem():
    counter = 0
    def __init__(self,id=None):
        
        self.G_dict = {}
        self.e_dict = {}
        self.GGT_dict = {}
        self.interface_local_dofs = {}
        self.local_kernel_dofs = {}
        self.local2global_alpha_dofs = {}
        self.global2local_alpha_dofs = {}
        self.local2global_lambda_dofs = {}
        self.global2local_lambda_dofs = {}
        
        if id is None:
            self.id = CourseProblem.counter
        else:
            self.id = id
        CourseProblem.counter +=1 

    def update_e_dict(self,local_e_dict):
        self.e_dict.update(local_e_dict)

    def update_G_dict(self,local_G_dict):
        self.G_dict.update(local_G_dict)

    def update_GGT_dict(self,local_GGT_dict):
        self.GGT_dict.update(local_GGT_dict)
    
    def compute_local_GGT_inv(self):
        pass
    
    def assemble_block_matrix(self,M_dict,row_map_dict,column_map_dict,shape):
        M = np.zeros(shape)
        for row_key, row_dofs in row_map_dict.items():
            for col_key, column_dofs in column_map_dict.items():
                if isinstance(col_key,int):
                    column_key = col_key
                else:
                    column_key = col_key[1]
                    if col_key[0]!=row_key:
                        continue

                M[np.ix_(row_dofs,column_dofs)] += M_dict[row_key,column_key]
        return M
            
    def assemble_block_vector(self,v_dict,map_dict,length):
        v = np.zeros(length)
        for row_keys, row_dofs in map_dict.items():
            v[np.ix_(row_dofs)] += v_dict[row_keys]
        return v

    def assemble_GGT(self,map_dict,shape):
        return self.assemble_block_matrix(self.GGT_dict,map_dict,map_dict,shape)
        
    def assemble_G(self,row_map_dict,column_map_dict,shape):
        return self.assemble_block_matrix(self.G_dict,row_map_dict,column_map_dict,shape)

    def assemble_e(self,map_dict,length):
        return self.assemble_block_vector(self.e_dict,map_dict,length)

class Solution():
    def __init__(self,u_dict, lambda_dict, alpha_dict,rk=None, proj_r_hist=None, lambda_hist=None):
        self.u_dict = u_dict 
        self.lambda_dict=lambda_dict 
        self.alpha_dict=alpha_dict
        self.rk=rk 
        self.proj_r_hist=proj_r_hist, 
        self.lambda_hist=lambda_hist

        
