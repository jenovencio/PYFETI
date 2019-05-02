import sys
import numpy as np
import scipy
import pandas as pd
import os
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csc_matrix, issparse
from unittest import TestCase, main
import logging
import time
import subprocess
import shutil

sys.path.append('../..')
from pyfeti.src.utils import OrderedSet, Get_dofs, save_object, load_object, pyfeti_dir, MPILauncher
from pyfeti.src.linalg import Matrix, Vector, elimination_matrix_from_map_dofs, \
                              expansion_matrix_from_map_dofs, ProjLinearSys, ProjPrecondLinearSys
from pyfeti.src import solvers


# geting path of MPI executable
mpi_exec = 'mpiexec'
try:
    mpi_path = os.environ['MPIDIR']
    mpi_exec = os.path.join(mpi_path, mpi_exec).replace('"','')
except:
    print("Warning! Using mpiexec in global path")
    mpi_path = None
    

try:
    python_path = os.environ['PYTHON_ENV']
    python_exec = os.path.join(python_path,'python').replace('"','')
except:
    print("Warning! Using python in global path")
    python_path = None
    python_exec = 'python'

def vector2localdict(v,map_dict):
    v_dict = {}
    for global_index, local_info in map_dict.items():
        for interface_id in local_info:
            v_dict[interface_id] = v[np.ix_(global_index)]

    return v_dict

class FETIsolver():
    def __init__(self,K_dict,B_dict,f_dict,**kwargs):
        self.K_dict = K_dict
        self.B_dict = B_dict
        self.f_dict = f_dict
        self.x_dict = None
        self.lambda_dict = None
        self.alpha_dict = None
        self.__dict__.update(kwargs)
        
        
    def solve(self):
        pass
        
    def serialize(self):
        for obj in [self.K_dict,self.B_dict,self.f_dict]:
            for key, item in obj:
                pass
                
class SerialFETIsolver(FETIsolver):
    def __init__(self,K_dict,B_dict,f_dict,**kwargs):
        super().__init__(K_dict,B_dict,f_dict,**kwargs)
        self.manager = SerialSolverManager(self.K_dict,self.B_dict,self.f_dict,**kwargs) 
        
    def solve(self):
       manager = self.manager

       start_time = time.time()
       manager.assemble_local_G_GGT_and_e()
       manager.assemble_cross_GGT()
       manager.build_local_to_global_mapping()
       build_local_matrix_time = time.time() - start_time

       G = manager.assemble_G()
       GGT = manager.assemble_GGT()
       e = manager.assemble_e()
       
       start_time = time.time()
       lambda_sol,alpha_sol, rk, proj_r_hist, lambda_hist = manager.solve_dual_interface_problem()
       elaspsed_time_PCPG = time.time() - start_time

       u_dict, lambda_dict, alpha_dict = manager.assemble_solution_dict(lambda_sol,alpha_sol)

       elapsed_time = time.time() - start_time
       return Solution(u_dict, lambda_dict, alpha_dict,rk, proj_r_hist, lambda_hist,
                       lambda_map=self.manager.local2global_lambda_dofs,alpha_map=self.manager.local2global_alpha_dofs,
                       u_map=self.manager.local2global_primal_dofs,lambda_size=self.manager.lambda_size,
                       alpha_size=self.manager.alpha_size,
                       solver_time=elapsed_time,local_matrix_time = build_local_matrix_time, 
                       time_PCPG = elaspsed_time_PCPG)
        
class SolverManager():
    def __init__(self,K_dict,B_dict,f_dict,pseudoinverse_kargs={'method':'svd','tolerance':1.0E-8},dual_interface_algorithm='PCPG',**kwargs):
        self.local_problem_dict = {}
        self.course_problem = CourseProblem()
        self.local2global_lambda_dofs = {}
        self.global2local_lambda_dofs = {}
        self.local2global_alpha_dofs = {}
        self.global2local_alpha_dofs = {}
        self.local2global_primal_dofs = {}
        self.global2local_primal_dofs = {}
        self.local_lambda_length_dict = {}
        self.local_alpha_length_dict = {}
        self.unique_map = {}
        self.local_problem_id_list = []
        self.lambda_size = None
        self.alpha_size = None
        self.primal_size = None
        self.e_dict = {}
        self.G = None
        self.e = None
        self.GGT = None
        self.num_partitions =  len(K_dict.keys())
        self.map_dofs = None
        self.pseudoinverse_kargs = pseudoinverse_kargs
        self.dual_interface_algorithm = dual_interface_algorithm
        self.is_local_G_GGT_and_e_computed = False
        
        # transform key args in object variables
        self.__dict__.update(kwargs)
        self.kwargs = kwargs

        self._create_local_problems(K_dict,B_dict,f_dict)
        
    @property
    def GGT_inv(self):
        return self.course_problem.compute_GGT_inv()

    def _create_local_problems(self,K_dict,B_dict,f_dict):
        
        for key, obj in K_dict.items():
            B_local_dict = B_dict[key]
            self.local_problem_id_list.append(key)
            self.local_problem_dict[key] = LocalProblem(obj,B_local_dict,f_dict[key],id=key,pseudoinverse_kargs=self.pseudoinverse_kargs)
            for interface_id, B in B_local_dict.items():
                self.local_lambda_length_dict[interface_id] = B.shape[0]
        
        self.local_problem_id_list.sort()

    def assemble_local_G_GGT_and_e(self):
        
        for problem_id, local_problem in self.local_problem_dict.items():
            R = local_problem.get_kernel()
            if R.shape[0]>0:
                self.e_dict[problem_id] = -R.T.dot(local_problem.f_local.data)
                self.course_problem.update_e_dict(self.e_dict)
                self.local_alpha_length_dict[problem_id] = R.shape[1]
                G_local_dict = {}
                GGT_local_dict = {}
                for key, B_local in local_problem.B_local.items():
                    local_id, nei_id = key
                    G = (-B_local.dot(R)).T
                    G_local_dict[key] = G
                    try:
                        GGT_local_dict[local_id,local_id] += G.dot(G.T)
                    except:
                        GGT_local_dict[local_id,local_id] = G.dot(G.T)

                self.course_problem.update_G_dict(G_local_dict)
                self.course_problem.update_GGT_dict(GGT_local_dict)
        
        self.is_local_G_GGT_and_e_computed = True

    def assemble_cross_GGT(self):
        GGT_local_dict = {}
        for (local_i, nei_i) , Gi in self.course_problem.G_dict.items():
            for (local_j, nei_j), Gj in self.course_problem.G_dict.items():
                if local_i!=local_j and local_i==nei_j and nei_i==local_j:
                    try:
                        if Gi.shape[0]>0 and Gj.shape[0]>0:
                            GGT_local_dict[local_i,local_j] = Gi.dot(Gj.T)
                    except:        
                        pass
        self.course_problem.update_GGT_dict(GGT_local_dict)
                    
    def assemble_e(self):
        try:
            self.e = self.course_problem.assemble_e(self.local2global_alpha_dofs,self.alpha_size)
            return  self.e
        except:
            raise('Build local to global mapping before calling this function')

    def assemble_GGT(self):
        try:
            self.GGT = self.course_problem.assemble_GGT(self.local2global_alpha_dofs,(self.alpha_size ,self.alpha_size))
            return self.GGT
        except:
            raise('Build local to global mapping before calling this function')
    
    def assemble_G(self):
        try:
            self.G = self.course_problem.assemble_G(self.local2global_alpha_dofs,self.local2global_lambda_dofs,(self.alpha_size ,self.lambda_size))
            return self.G
        except:
            raise('Build local to global mapping before calling this function')
    
    def compute_lambda_im(self):
        return  self.G.T.dot(self.GGT_inv.dot(self.e))
        
    def get_vdot(self):
        return lambda v,w : np.dot(v,w)

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
                gap = u_dict[local_id,nei_id] + u_dict[nei_id,local_id]
                gap_dict[local_id, nei_id] = gap
                gap_dict[nei_id, local_id] = -gap
            elif nei_id==local_id:
                logging.warning('Dirichlet B.C set to 0!')
                gap = u_dict[local_id,nei_id] 
                gap_dict[local_id, nei_id] = gap

        return gap_dict

    def solve_interface_force(self,gap_dict,**kwargs):
        gap_f_dict = {}
        for problem_id, local_problem in self.local_problem_dict.items():
            f_dict_local = local_problem.apply_schur_complement(gap_dict,**kwargs)
            gap_f_dict.update(f_dict_local)


        # compute interface force avg
        f_dict = {}
        for interface_id in gap_dict:
            local_id, nei_id = interface_id
            if nei_id>local_id:
                gap = gap_f_dict[local_id,nei_id] + gap_f_dict[nei_id,local_id]
                f_dict [local_id, nei_id] = gap
                f_dict [nei_id, local_id] = gap
            elif nei_id==local_id:
                gap = u_dict[local_id,nei_id] 
                f_dict[local_id, nei_id] = gap


        return f_dict

    def solve_dual_interface_problem(self,algorithm=None):
        
        if algorithm is None:
            algorithm = self.dual_interface_algorithm

        lambda_im = self.compute_lambda_im()
        G = self.G
        GGT_inv = self.GGT_inv
        I = np.eye(self.lambda_size)
        Projection_action = lambda r : (I - G.T.dot(GGT_inv.dot(G))).dot(r)
        F_action = lambda lambda_ker : self.apply_F(lambda_ker)
        vdot = self.get_vdot()

        Precondicioner_action = None
        try:
            precond_type = self.precond_type
            if self.precond_type is not None:
                Precondicioner_action = lambda gap_u : self.apply_F_inv(gap_u,precond_type=precond_type )
        except:
            pass
            

        residual = -self.apply_F(lambda_im, external_force=True)
        d = -self.apply_F(0.0*lambda_im, external_force=True)
        norm_d = np.linalg.norm(d)

        logging.info('Dual Interface algorithm = %s' %algorithm)
        method_to_call = getattr(solvers, algorithm)

        try:
            tolerance = norm_d*self.tolerance 
        except:
            tolerance = None # using default tolerance of the choosen interface algorithm
           
        try:
            max_int = self.max_int
        except:
            max_int = None # using default max_int of the choosen interface algorithm

        lambda_ker, rk, proj_r_hist, lambda_hist = method_to_call(F_action,residual,Projection_action=Projection_action,
                                                         lambda_init=None,
                                                         Precondicioner_action=Precondicioner_action,
                                                         tolerance=tolerance,max_int=max_int,vdot=vdot)

        lambda_sol = lambda_im + lambda_ker
        logging.debug(('lambda_im=',lambda_im))
        logging.debug(('lambda_sol=',lambda_sol))

        alpha_sol = GGT_inv.dot(G.dot(residual - self.apply_F(lambda_ker)))

        return lambda_sol,alpha_sol, rk, proj_r_hist, lambda_hist

    def vector2localdict(self,v,map_dict):
        return vector2localdict(v,map_dict)

    def apply_F(self, v,  external_force=False):
       
        v_dict = self.vector2localdict(v, self.global2local_lambda_dofs)
        gap_dict = self.solve_interface_gap(v_dict,external_force)
        
        d = np.zeros(self.lambda_size)
        for interface_id,global_index in self.local2global_lambda_dofs.items():
            d[global_index] += gap_dict[interface_id]

        return -d

    def apply_F_inv(self,v,**kwargs):
        # map array to domains dict and then solve force gap
        v_dict = self.vector2localdict(v, self.global2local_lambda_dofs)
        gap_dict = self.solve_interface_force(v_dict,**kwargs)

        # assemble vector based on dict
        d = np.zeros(self.lambda_size)
        for interface_id,global_index in self.local2global_lambda_dofs.items():
            d[global_index] += gap_dict[interface_id]

        return d

    def build_local_to_global_mapping(self):
        ''' This method create maps between local domain and the global
        domain, where the Global is NOT composed by the unique set of a variables.
     
        The mapping dictionaries are create for the primal variable (displacement)
        lambdas, and alphas, where :

        Ku = f + B*lambda + R*alpha

        u -> set of NOT unique primal variables
        lambda -> set of interface compatibilite multipliers
        alpha -> a set correction multipliers based on kernel space (R)

        u = u_dual = [u1, u2] -> global set of variables
        
        '''

        if  not self.is_local_G_GGT_and_e_computed:
            self.assemble_local_G_GGT_and_e()

    
        dof_primal_init = 0
        dof_lambda_init = 0
        dof_alpha_init = 0
        for local_id in self.local_problem_id_list:
            local_problem = self.local_problem_dict[local_id]
            local_primal_dofs = np.arange(local_problem.length) 
            global_primal_index = dof_primal_init + local_primal_dofs
            dof_primal_init+= local_problem.length
            self.local2global_primal_dofs[local_id] = global_primal_index
            self.global2local_primal_dofs[tuple(global_primal_index)] = {local_id:local_primal_dofs}

            if local_id in self.local_alpha_length_dict:
                local_alpha_length = self.local_alpha_length_dict[local_id]
                local_alpha_dofs = np.arange(local_alpha_length) 
                global_alpha_index = dof_alpha_init + local_alpha_dofs
                dof_alpha_init+=local_alpha_length
                self.local2global_alpha_dofs[local_id] = global_alpha_index
                self.global2local_alpha_dofs[tuple(global_alpha_index)] = {local_id:local_alpha_dofs}
                    
            for nei_id in local_problem.neighbors_id:
                # nei_id==local_id represents non-physical domains, e.g. Boundary conditions
                if nei_id>=local_id:
                    try:
                        local_lambda_length = self.local_lambda_length_dict[local_id,nei_id]
                        local_dofs = np.arange(local_lambda_length) 
                        global_index = dof_lambda_init + local_dofs
                        dof_lambda_init+=local_lambda_length
                        self.local2global_lambda_dofs[local_id,nei_id] = global_index 
                        self.global2local_lambda_dofs[tuple(global_index)] = {(local_id,nei_id):local_dofs}
                        B_indices = local_problem.B_local[local_id,nei_id].nonzero()

                    except:
                        continue
                else:
                    B_indices = local_problem.B_local[local_id,nei_id].nonzero()

                 
                self.unique_map[local_id,nei_id] = self.local2global_primal_dofs[local_id][B_indices[1]]

        self.lambda_size = dof_lambda_init
        self.alpha_size = dof_alpha_init
        self.primal_size = dof_primal_init

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

    def assemble_global_B(self):
        ''' This method assembles the global B matrix, based on local 
        B matrices.

        B_global = [B_local(1,2)  - B_local(2,1) ]

        '''
        try:
            B = scipy.sparse.lil_matrix((self.lambda_size,self.primal_size),dtype=np.int)
        except:
            self.build_local_to_global_mapping()
            B = scipy.sparse.lil_matrix((self.lambda_size,self.primal_size),dtype=np.int)

        for local_id in self.local_problem_id_list:
            local_problem = self.local_problem_dict[local_id]
            Bi_dict = local_problem.B_local
            idx = self.local2global_primal_dofs[local_id]
            for nei_id in local_problem.neighbors_id:
                if local_id < nei_id:
                    idy =  self.local2global_lambda_dofs[local_id, nei_id]
                else:
                    idy =  self.local2global_lambda_dofs[nei_id,local_id]

                Bij = Bi_dict[local_id, nei_id]
                #B[np.ix_(idy,idx)] = np.sign(nei_id - local_id)*Bij
                B[np.ix_(idy,idx)] = Bij

        return B.tocsc()

    def build_dof_map(self):
        
        
        if not self.unique_map:
            self.build_local_to_global_mapping()

        map_dofs = self.build_global_map_dataframe()

        non_uniform_map = list(map_dofs['Global_dof_id'])
        removed_dofs = []
        kept_dofs = []
        for local_id,nei_id in self.unique_map:
            if (nei_id,local_id) in self.unique_map and nei_id>local_id:
                for unique_dof,duplicated_dof in zip(self.unique_map[local_id,nei_id],self.unique_map[nei_id,local_id]):
                    
                    if unique_dof not in removed_dofs:
                        non_uniform_map[duplicated_dof] = unique_dof
                    else:
                        unique_dof = kept_dofs[removed_dofs.index(unique_dof)]
                        non_uniform_map[duplicated_dof] = unique_dof

                    removed_dofs.append(duplicated_dof)
                    kept_dofs.append(unique_dof)

        #primal_dofs = np.arange(self.primal_size - self.lambda_size)
        unique_list = list(set(non_uniform_map))
        primal_dofs = np.arange(len(unique_list))
        uniform_map = lambda global_id : primal_dofs[unique_list.index(global_id)]
        non_uni_2_uni = []
        for nun_uniform_id in non_uniform_map:
            non_uni_2_uni.append(uniform_map(nun_uniform_id))

        map_dofs['Primal_dof_id'] = non_uni_2_uni
        self.map_dofs = map_dofs
        return self.map_dofs
    
    def assemble_global_L(self):
        if self.map_dofs is None:
            self.build_dof_map()
        map_dofs = self.map_dofs

        return elimination_matrix_from_map_dofs(map_dofs,primal_tag='Primal_dof_id')

    def assemble_global_L_exp(self):
        if self.map_dofs is None:
            self.build_dof_map()
        map_dofs = self.map_dofs
        return expansion_matrix_from_map_dofs(map_dofs,primal_tag='Primal_dof_id')

    def build_global_map_dataframe(self):
        list_domain = []
        list_local_dof = []
        list_global_dof = []
        for domain_id in self.local2global_primal_dofs:   
            global_dof_id = list(self.local2global_primal_dofs[domain_id])
            local_dof_id = list(self.global2local_primal_dofs[tuple(global_dof_id)][domain_id])
            list_domain.extend([domain_id]*len(local_dof_id))
            list_local_dof.extend(local_dof_id)
            list_global_dof.extend(global_dof_id)

        map_dict = {}
        map_dict['Domain_id'] = list_domain
        map_dict['Local_dof_id'] = list_local_dof
        map_dict['Global_dof_id'] = list_global_dof
        
        return pd.DataFrame.from_dict(map_dict)

    def assemble_global_K_and_f(self):

        K_list = []
        fext_list = []
        for local_id in self.local_problem_id_list:
            K_list.append(self.local_problem_dict[local_id].K_local.data)
            fext_list.append(self.local_problem_dict[local_id].f_local.data)

        Kd = scipy.sparse.block_diag(K_list)
        fd = np.hstack(fext_list)
        
        self.block_stiffness = Kd
        self.block_force = fd

        return Kd.tocsc(), fd 

    def assemble_global_kernel(self):
        ''' This method assembles the global R matrix, based on local 
        R matrices.

        R_global = [R_local(1),  0         ]  alpha(1) 
                   [     0      R_local(2) ]  alpha(2) 

        '''

        if not self.local2global_alpha_dofs:
            self.build_local_to_global_mapping()

        try:
            R = scipy.sparse.lil_matrix((self.primal_size,self.alpha_size))
        except:
            self.build_local_to_global_mapping()
            R = scipy.sparse.lil_matrix((self.alpha_size,self.primal_size))

        for local_id in self.local_problem_id_list:
            local_problem = self.local_problem_dict[local_id]
            
            
            idx = self.local2global_primal_dofs[local_id]
            try:
                idy =  self.local2global_alpha_dofs[local_id]
                R_local = local_problem.kernel
                R[np.ix_(idx,idy)] = R_local
            except:
                continue

        return R.tocsc()

    def assemble_global_F(self):
        ''' This function return F as a linear operator
        '''

        F_action = lambda lambda_ker : self.apply_F(lambda_ker)
        return LinearOperator((self.lambda_size,self.lambda_size), matvec=F_action)  

    def assemble_global_d(self):
        ''' This function assembles the d vector related to the followinf system


        $$
        \begin{bmatrix} F & G^{T} \\
                         G & 0  
        \end{bmatrix}
        \begin{bmatrix} \lambda  \\ 
        \alpha
        \end{bmatrix}
        =
        \begin{bmatrix} d \\ 
        e \end{bmatrix}
        $$
        '''
        return -self.apply_F(np.array(self.lambda_size*[0.0]), external_force=True)


class ParallelSolverManager(SolverManager):
    def __init__(self,K_dict,B_dict,f_dict,pseudoinverse_kargs={'method':'svd','tolerance':1.0E-8},temp_folder='temp',**kwargs):
        self.temp_folder = temp_folder
        self.local_problem_path = {}
        self.prefix = 'local_problem_'
        self.ext = '.pkl'
        self.log = True
        super().__init__(K_dict,B_dict,f_dict,pseudoinverse_kargs=pseudoinverse_kargs,**kwargs)
        
    def _create_local_problems(self,K_dict,B_dict,f_dict,temp_folder=None):
        start_time = time.time()
        if temp_folder is None:
            temp_folder = self.temp_folder
        else:
            self.temp_folder = temp_folder

        try:
            #deleting local files
            shutil.rmtree(temp_folder, ignore_errors=True)
        except:
            pass

        try:
            # creating folder for MPI execution
            os.mkdir(temp_folder)
        except:
            pass
            
        for key, obj in K_dict.items():
            B_local_dict = B_dict[key]
            self.local_problem_id_list.append(key)
            self.local_problem_dict[key] = LocalProblem(obj,B_local_dict,f_dict[key],id=key,pseudoinverse_kargs=self.pseudoinverse_kargs)
            for interface_id, B in B_local_dict.items():
                self.local_lambda_length_dict[interface_id] = B.shape[0]

            local_path =  os.path.join(temp_folder, self.prefix + str(key) + self.ext)
            self.local_problem_path[key] = local_path
            save_object(self.local_problem_dict[key] , local_path)

        elapsed_time = time.time() - start_time
        logging.info('{"serializing_local_problems" : %4.5e} #Elapsed time (s)' %elapsed_time)

    def launch_mpi_process(self):
        python_file = pyfeti_dir(os.path.join('src','MPIsolver.py'))

        start_time = time.time()
        
        mpi_obj = MPILauncher(python_file,
                              mpi_size=self.num_partitions,
                              module = 'MPIsolver',
                              method = 'launch_ParallelSolver',
                              tmp_folder=self.temp_folder ,
                              prefix = self.prefix, 
                              ext = self.ext,
                              **self.kwargs)
        
        elapsed_time = time.time() - start_time
        logging.info('{"mpi_launcher" : %f} #Elapsed time (s)' %elapsed_time)

        start_time = time.time()
        localtime = time.asctime( time.localtime(time.time()) )
        logging.info('Local Time before mpi run: %s' %localtime)
        mpi_obj.run()
        elapsed_time = time.time() - start_time
        logging.info('{"mpi_run" : %f} #Elapsed time (s)' %elapsed_time)
        localtime = time.asctime( time.localtime(time.time()))
        logging.info('Local Time after mpi run: %s' %localtime)

    def read_results(self):
        logging.info('Reading results from MPISolver')
        start_time = time.time()
        solution_path = os.path.join(self.temp_folder,'solution.pkl')
        u_dict = {}
        alpha_dict = {}
        for i in range(1,self.num_partitions+1):
            try:
                displacement_path = os.path.join(self.temp_folder,'displacement_' + str(i) + '.pkl')
                alpha_path = os.path.join(self.temp_folder,'alpha_' + str(i) + '.pkl')                
                u_dict[i] =  load_object(displacement_path)
                alpha_dict[i] =  load_object(alpha_path,tries=1,sleep_delay=0)
            except:
                pass
            
            
        sol_obj = load_object(solution_path)
        sol_obj.u_dict = u_dict
        sol_obj.alpha_dict = alpha_dict

        elapsed_time = time.time() - start_time
        logging.info('{ "load_results": %f} #Elapsed time (s)' %elapsed_time)
        return sol_obj

    def delete(self):
        shutil.rmtree(self.temp_folder)


class ParallelFETIsolver(FETIsolver):
    def __init__(self,K_dict,B_dict,f_dict,temp_folder='temp',delete_folder=False,**kwargs):
        super().__init__(K_dict,B_dict,f_dict,**kwargs)
        self.manager = ParallelSolverManager(self.K_dict,self.B_dict,self.f_dict,temp_folder=temp_folder,**kwargs) 
        self.delete_folder = delete_folder

    def solve(self):
        logging.info('')
        manager = self.manager        
        manager.launch_mpi_process()
        sol_obj = manager.read_results()

        if self.delete_folder:
            manager.delete()

        return sol_obj
        

class LocalProblem():
    counter = 0
    def __init__(self,K_local, B_local, f_local,id,pseudoinverse_kargs={'method':'svd','tolerance':1.0E-8}):
        LocalProblem.counter+=1
        if isinstance(K_local,Matrix):
            self.K_local = K_local
        else:
            self.K_local = Matrix(K_local,pseudoinverse_kargs=pseudoinverse_kargs)

        self.length = self.K_local.shape[0]
        if isinstance(f_local,Vector):
            self.f_local = f_local
        else:
            self.f_local = Vector(f_local)

        self.B_local = B_local
        self.solution = None
        
        self.id = id
        self.interface_size =  0
        self.neighbors_id = []
        self.crosspoints = {}
        self.interface_set = set()
        self.interior_set = set()
        self.scalling = None
        self.get_neighbors_id()
        self.compute_interface_dof_set()
        self.compute_interior_dof_set()
        self.compute_neighbor_scalling_array()

    def get_neighbors_id(self):
        for nei_id, obj in self.B_local.items():
            self.neighbors_id.append(nei_id[1])
            self.interface_size += obj.shape[0]
        self.neighbors_id.sort()

    def compute_interface_dof_set(self):
        ''' This function compute the set of dofs at the 
        interface based on local B matrices
        '''
        for key, B in self.B_local.items():
            self.interface_set.update(B.nonzero()[1])
            
        return self.interface_set

    def compute_interior_dof_set(self):
        ''' This function compute the set of dofs at the 
        interior based on interface set dofs
        '''
        if not self.interface_set:
            self.compute_interface_dof_set()
        
        self.interior_set.update(set(list(range(self.length)))-self.interface_set)
        return self.interior_set

    def compute_neighbor_scalling_array(self):
        ''' This method compute an array with dimension
        equal to the self.length size, and values equal 
        to number of neighbors + 1
        '''
        
        scalling = np.ones(self.length, dtype=np.int)
        for key, B in self.B_local.items():
            scalling[B.nonzero()[1]] += 1 

        self.scalling = scalling
        return self.scalling

    def expand_interface_gap(self,gap_dict):
        ''' This method expands a u gap_dict given at the interface
        pairs to the whole domain, named [ub, ui] based on 
        scalling B.T expation.
        
        u = scalling sum [B i u ij]

        ub_dict = self.get_interface(u)

        returns:
            u : np.array
                array with primal variables 
        '''
        u = np.zeros(self.length)
        for interface_id, B in self.B_local.items():
                (local_id,nei_id) = interface_id
                if local_id>nei_id:
                    interface_id = (nei_id,local_id) 
                u += B.T.dot(gap_dict[interface_id])

        # scalling gap
        u = sparse.diags(1.0/self.scalling).dot(u)
        
        return u

    def apply_schur_complement(self,gap_dict,precond_type='Lumped'):
        ''' This method computes the force at the interface, 
        such that produces the gap at the interface given by te gap_dict
        which is defined as the primal variable at the interface given
        the interface pair dictionary
        '''

        
        interface_id = list(self.interface_set)
        interior_id = list(self.interior_set)
        f = np.zeros(self.length)
        u = self.expand_interface_gap(gap_dict)
        ub = u[interface_id]
        K = self.K_local.data
        Kbb = K[np.ix_(interface_id,interface_id)]
        if precond_type=='Lumped':
            f[interface_id] += Kbb.dot(ub)

        elif precond_type=='SuperLumped':
            f[interface_id] += np.diag(Kbb.diagonal()).dot(ub)

        elif precond_type=='LumpedDirichlet':
            Kib = self.K_local.data[np.ix_(interior_id,interface_id)]
            Kii_inv = np.diag(1.0/(K[np.ix_(interior_id,interior_id)].diagonal()))
            f_exp = Kib.dot(ub)
            ui = Kii_inv.dot(f_exp)
            f[interface_id] += Kbb.dot(ub) - Kib.T.dot(ui)
        
        elif precond_type=='Dirichlet':
            
            Kib = self.K_local.data[np.ix_(interior_id,interface_id)]
            f_exp = np.zeros(self.length)
            f_exp[interior_id] += Kib.dot(ub)
            Kii = K[np.ix_(interior_id,interior_id)]
            ui = sparse.linalg.spsolve(Kii,f_exp[interior_id])
            #ui = self.K_local.apply_inverse(f_exp)[interior_id]
            f[interface_id] += Kbb.dot(ub) - Kib.T.dot(ui)

        else:
            logging.error('Schur complement type not supported!')

        
        f_scalling = sparse.diags(1.0/self.scalling).dot(f)

        return self.get_interface_dict(f_scalling)

    def crosspoints_dectection(self):
        ''' This function detects cross points based on local 
        information, crosspoints are defined as tuples with more 
        the two intries, e.g. (1,2,3) a node is connected to 3 domains
        but it can appears in an another crosspoint tuple.
        The crosspoints tuple points work as a key to the pointers of local B matrix:

        crosspoints[i,j,k] = (dof_u, dof_lambda_j, dof_lambda_k)

        where dof_u is a columns id of the Local B matrices, 
        and dof_lambda_j, dof_lambda_k are the rows of the local B matrices:

        Bij [dof_lambda_j, :] -> is associated with lambda_ij
        Bik [dof_lambda_k, :] -> is associated with lambda_ik

        these line represents a redudant constraint in the global system

        Returns
            crosspoints : dict
                a dict with key as crosspoint tuple, and values points to 
                contraint equation in the local B matrix

        '''
        crosspoints = self.crosspoints
        for nei_id_j in self.neighbors_id:
            for nei_id_k in self.neighbors_id:
                if nei_id_k>nei_id_j:
                    Bij = np.abs(self.B_local[self.id,nei_id_j])
                    Bik = np.abs(self.B_local[self.id,nei_id_k])
                    try:
                        B_stack = sparse.vstack((Bij,Bik))
                    except:
                        B_stack = np.vstack((Bij,Bik))

                    try:
                        col_id = np.argwhere(B_stack.T.sum(axis=1)>1)[0][0]
                        row_id_j = np.argwhere(Bij[:,col_id]>0)[0][0] 
                        row_id_k = np.argwhere(Bik[:,col_id]>0)[0][0] 
                        crosspoints[self.id,nei_id_j,nei_id_k] = {col_id : (row_id_j,row_id_k)}
                        logging.info('Crosspoint detected in Domain id = %i, dofs is connected to domains (%i and %i)' %(self.id,row_id_j,row_id_k))
                    except:
                        continue

                else:
                    continue

        return crosspoints

    def solve(self, lambda_dict,external_force_bool=False):
       
        if not external_force_bool:
            f = np.zeros(self.f_local.data.shape)
        else:
            f = np.copy(self.f_local.data)

        if lambda_dict is not None:
            # assemble interface and external forces
            #for interface_id, f_interface in lambda_dict.items():
            for interface_id, B in self.B_local.items():
                (local_id,nei_id) = interface_id
                if local_id>nei_id:
                    interface_id = (nei_id,local_id) 
                f -= B.T.dot(lambda_dict[interface_id])
        return self.K_local.apply_inverse(f)

    def get_interface_dict(self,x):
        interface_dict = {}
        for interface_id, Bij in self.B_local.items():
            (local_id,nei_id) = interface_id
            #interface_dict[interface_id] = (nei_id-local_id)*Bij.dot(x)
            interface_dict[interface_id] = Bij.dot(x)

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
        self.GGT = None
        self.GGT_inv = None
        self.course_method = 'inv'
     
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
    
    def compute_GGT_inv(self,course_method=None,**kwargs):

        if self.GGT_inv is None:
            if course_method is None:
                course_method = self.course_method

            if course_method == 'splu':
                GGT = scipy.sparse.matrix.csr(self.GGT)
                self.GGT_inv.dot = lambda x : scipy.sparse.linalg.splu(GGT).solve 
            elif course_method == 'inv':
                self.GGT_inv = np.linalg.inv(self.GGT)
            else:
                raise('G@G.T is not defined.')

        return self.GGT_inv
    
    def compute_local_GGT_columns_inv(self,columns_id=None):

        n = self.GGT.shapep[0]
        GGT_inv_columns = np.array([],shape=(n,len(columns_id)))
        if columns_id is not None:
            I = np.identity()
            b = I[:columns_id]
            GGT_inv_columns = self.GGT_inv.dot(b)
        return GGT_inv_columns
            
    def assemble_block_matrix(self,M_dict,row_map_dict,column_map_dict,shape):
        M = np.zeros(shape)
        for row_key, row_dofs in row_map_dict.items():
            for col_key, column_dofs in column_map_dict.items():
                if isinstance(col_key,int):
                    column_key = col_key
                else:
                    if row_key not in col_key:
                        continue
                    else:
                        l_key = list(col_key)
                        l_key.remove(row_key)
                        column_key = l_key[0]
                try:        
                    M[np.ix_(row_dofs,column_dofs)] += M_dict[row_key,column_key]
                except:
                    continue

        return M
            
    def assemble_block_vector(self,v_dict,map_dict,length):
        v = np.zeros(length)
        for row_keys, row_dofs in map_dict.items():
            v[np.ix_(row_dofs)] += v_dict[row_keys]
        return v

    def assemble_GGT(self,map_dict,shape):
        self.GGT = self.assemble_block_matrix(self.GGT_dict,map_dict,map_dict,shape) 
        return self.GGT
        
    def assemble_G(self,row_map_dict,column_map_dict,shape):
        return self.assemble_block_matrix(self.G_dict,row_map_dict,column_map_dict,shape)

    def assemble_e(self,map_dict,length):
        return self.assemble_block_vector(self.e_dict,map_dict,length)


class Solution():
    def __init__(self,u_dict, lambda_dict, alpha_dict,rk=None, proj_r_hist=None, lambda_hist=None, 
                 lambda_map=None,alpha_map=None,u_map=None,lambda_size=None,alpha_size=None,**kwargs):
        
        self.lambda_dict=lambda_dict 
        self.alpha_dict=alpha_dict
        self.rk=rk 
        self.proj_r_hist=proj_r_hist 
        self.lambda_hist=lambda_hist
        self.lambda_map = lambda_map
        self.alpha_map = alpha_map
        self.u_map = u_map
        self.lambda_size =lambda_size
        self.alpha_size = alpha_size
        self._rebuild_lambda_map()
        self.domain_list = None
        self.u_dict = u_dict
        self.__dict__.update(kwargs)

    @property
    def u_dict(self):
        return self._u_dict 

    @u_dict.setter
    def u_dict(self,u_dict):
        self._u_dict = u_dict
        self.domain_list = np.sort(list(u_dict.keys()))

    @property
    def PCGP_iterations(self):
        return len(self.proj_r_hist)

    @property
    def projected_residual(self):
        try:
            return self.proj_r_hist[-1]
        except:
            return 0

    def _rebuild_lambda_map(self):

        local_dict ={}
        try:
            for key, interface_dict in self.lambda_dict.items():
                for (dom_id,nei_id),values in interface_dict.items():
                    if nei_id>=dom_id:
                         local_dict[dom_id,nei_id] = values
            self.lambda_dict = local_dict
        except:
            pass

    @property
    def displacement(self):
        u = np.array([])
        for key in self.domain_list:
            u = np.append(u,self.u_dict[key])
        
        return u
    
    @property
    def interface_lambda(self):
        ''' assemble lambda in the interface
        '''
        return self.assemble_vector(self.lambda_dict,self.lambda_size,self.lambda_map)

    @property
    def alpha(self):
        ''' assemble lambda in the interface
        '''
        return self.assemble_vector(self.alpha_dict,self.alpha_size,self.alpha_map)

    def assemble_vector(self,v_dict,vector_length,map_dict):
        v = np.zeros(vector_length)
        for map_index,row_dofs in map_dict.items():
            v[np.ix_(row_dofs)] += v_dict[map_index]
        return v


#Alias variables for backward and future compatibilite
SerialSolverManager = SolverManager
FETIManager = SerialSolverManager


def cyclic_eig(K_dict,M_dict,B_dict,f_dict,num_of_modes=20,use_precond=True):
    ''' This method compute the cyclic eigenvalues and eigenvector of
    the block Hybrid eigenvalue problem.

    Parameters:
        K_dict : dict
            dictionary with block stiffeness matrix connected to the sector index (dict key)
        M_dict : dict
            dictionary with block mass matrix connected to the sector index (dict key)
        B_dict : dict
            dictionary with block Boolean matrix connected to the sector index (dict key)
        f_dict : dict
            dictionaty with block forces which are 0. The dict is not used for the eigen computation

    returns:
        frequency : list
            frequency in Hertz
        
        modes_dict: dict
            dictionary with the eingenvectors
        info_dict : dict
            dictionary with solve info

    '''
    info_dict = {}
    info_dict['Start'] = time.time()


    feticase = FETIManager(K_dict,B_dict,f_dict)
    feticase2 = FETIManager(M_dict,B_dict,f_dict)
    K_global, f_global = feticase.assemble_global_K_and_f()
    M_global, f_global = feticase2.assemble_global_K_and_f()
    B = feticase.assemble_global_B()
    ndofs = K_global.shape[0]
    I = scipy.sparse.eye(ndofs)
    # creating projection
    P = scipy.sparse.eye(K_global.shape[0]) - 0.5*B.T.dot(B)
    if use_precond:
        pre_obj = ProjPrecondLinearSys(K_global, P, incomplete=False )
        Precond = pre_obj.getLinearOperator()
        lo_obj = ProjLinearSys(K_global,M_global,P,Precond, linear_solver=None)
    else:
        lo_obj = ProjLinearSys(K_global,M_global,P)

    Dp_new = lo_obj.getLinearOperator()
    w2new, v2new = scipy.sparse.linalg.eigs(Dp_new, k=num_of_modes)
    new_id = np.argsort(w2new)[::-1]
    w2new = w2new[new_id]
    v2new = v2new[:,new_id]
    eigenvalues = 1.0/(w2new.real)
    omega = np.sqrt(eigenvalues)
    frequency = omega/(2.0*np.pi)

    local_dofs = K_dict[0].shape[0]
    modes_dict = {}
    
    for key in f_dict:
        modes_dict[key] = v2new[key*local_dofs:(key+1)*local_dofs,:]

    info_dict['TotalLienarOpCall'] = lo_obj.num_iters
    info_dict['AvgLienarOpCall'] = lo_obj.num_iters/lo_obj.solver_counter
    info_dict['SolverCalls'] = lo_obj.solver_counter
    info_dict['End'] = time.time()
    info_dict['Time'] = info_dict['End'] - info_dict['Start'] 

    return frequency, modes_dict, info_dict


class  Test_FETIsolver(TestCase):
    def test_crosspoints_dectection(self):

        K1 = np.array([[4., 0., 0., 0.],
                       [0., 4., -1., -2.],
                       [0., -1., 4., -1.],
                       [0., -2., -1., 4.]])

        K2 = K3 = K4 = np.array([[4., -1., -2., -1.],
                                 [-1., 4., -1., -2.],
                                 [-2., -1., 4., -1.],
                                 [-1., -2., -1., 4.]])


        q0 = 10.0
        q1 = np.array([0.,0.,0.,0.])
        q2 = np.array([0.,0.,0.,0.])
        q3 = np.array([0.,0.,0.,0.])
        q4 = np.array([0.,0.,1.0,0.0])

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


        # Using PyFETI to solve the probrem described above
        K1 = sparse.csc_matrix(K1)
        K2 = sparse.csc_matrix(K2)
        K_dict = {1:K1,2:K2, 3:K2, 4:K2}
        B_dict = {1 : {(1,2) : B12, (1,3) : B13, (1,4) : B14}, 
                  2 : {(2,1) : B21, (2,4) : B24,(2,3) : B23}, 
                  3 : {(3,1) : B31, (3,4) : B34, (3,2) : B32}, 
                  4 : {(4,2) : B42, (4,3) : B43, (4,1) : B41}}

        q_dict = {1:q1 ,2:q2, 3:q3, 4:q4}

        #solver_obj = FETIManager(K_dict,B_dict,q_dict)

        #nc = FETIManager.lambda_size
        #solver_obj.apply_F(np.ones(nc))

        crosspoints_global_dict = {}

        interface_dofs_target = {1: [1,2,3],
                                 2 : [0,2,3],
                                 3 : [0,1,2],
                                 4 : [0,1,3]}
        
        interior_dofs_target = {1: [0],
                                 2 : [1],
                                 3 : [3],
                                 4 : [2]}


        import copy   
        gap_dict = copy.deepcopy(B_dict)
        for i,B in gap_dict.items():
            for key, item in B.items():
                B[key] = np.ones(item.shape[0])

        for i in range(1,5):
            local_obj = LocalProblem(K_dict[i], B_dict[i],q_dict[i],id=i)
            interface_dofs = local_obj.compute_interface_dof_set()
            np.testing.assert_equal(interface_dofs_target[i], list(interface_dofs))

            interior_dofs = local_obj.compute_interior_dof_set()
            np.testing.assert_equal(interior_dofs_target[i], list(interior_dofs))

            scalling = local_obj.compute_neighbor_scalling_array()
            #u =  local_obj.expand_interface_gap(gap_dict[i])
            force_dict = local_obj.apply_schur_complement(gap_dict[i])
            force_dict = local_obj.apply_schur_complement(gap_dict[i],precond_type='SuperLumped')
            force_dict = local_obj.apply_schur_complement(gap_dict[i],precond_type='Dirichlet')
            force_dict = local_obj.apply_schur_complement(gap_dict[i],precond_type='LumpedDirichlet')
            
            crosspoints_global_dict[i] = local_obj.crosspoints_dectection()

        crosspoints_global_dict


        # test expansion

        



if __name__=='__main__':

    main()