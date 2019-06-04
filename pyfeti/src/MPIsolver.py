# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 08:35:26 2017

@author: ge72tih
"""

        

import subprocess
import sys
import dill as pickle
import numpy as np
import scipy.sparse as sparse
import scipy
import logging
import time

from pyfeti.src.utils import save_object, load_object, Log, getattr_mpi_attributes
from pyfeti.src.feti_solver import CoarseProblem, Solution, SolverManager, vector2localdict
from pyfeti.src import solvers
from pyfeti.src.MPIlinalg import exchange_info, exchange_global_dict, pardot, RetangularLinearOperator, ParallelRetangularLinearOperator

from mpi4py import MPI
import os


class ParallelSolver(SolverManager):
    def __init__(self,obj_id, local_problem, **kwargs):
        

        self.obj_id = obj_id
        self.local_problem = local_problem

        self.residual = []
        self.lampda_im = []
        self.lampda_ker = []
        
        self.course_problem = CoarseProblem(obj_id)

        self.local2global_lambda_dofs = {}
        self.global2local_lambda_dofs = {}
        self.local2global_alpha_dofs = {}
        self.global2local_alpha_dofs = {}
        self.local2global_primal_dofs = {}
        self.global2local_primal_dofs = {}
        
        self.local_lambda_length_dict = {}
        self.local_alpha_length_dict = {}
        self.local_primal_length_dict = {}
        self.local_primal_length_dict[self.obj_id] = self.local_problem.length
        
        
        self.global_alpha_length_dict = {}
        self.global_lambda_length_dict = {}
        self.global_primal_length_dict = {}
         
        
        self.unique_map = {}
        self.local_problem_id_list = []
        self.lambda_size = None
        self.alpha_size = None
        self.primal_size = None
        self.e_dict = {}
        self.G = None
        self.e = None
        self.GGT = None
        self.GGT_dict = {}
        #self.GGT_inv = None
        self.num_partitions = comm.Get_size()
        self.partitions_list = list(range(1,self.num_partitions+1))
        self.neighbors_id = self.local_problem.neighbors_id
        self.dual_interface_algorithm = 'PCPG'
        # transform key args in object variables
        self.__dict__.update(kwargs)

        logging.info('local length = %i' %self.local_problem.length)
        
    def _exchange_global_size(self):
        local_id = self.obj_id
        for nei_id in self.local_problem.neighbors_id:
            interface_id = (local_id,nei_id)
            self.local_lambda_length_dict[interface_id] = self.local_problem.B_local[interface_id].shape[0]
        
        for global_id in self.partitions_list:
            if global_id!=local_id:
                lambda_nei_dict = exchange_info(self.local_lambda_length_dict,local_id,global_id)
                alpha_nei_dict = exchange_info(self.local_alpha_length_dict,local_id,global_id)
                primal_nei_dict = exchange_info(self.local_primal_length_dict,local_id,global_id)
                self.local_lambda_length_dict.update(lambda_nei_dict)
                self.local_alpha_length_dict.update(alpha_nei_dict)
                self.local_primal_length_dict.update(primal_nei_dict)

        return None
        
    def mpi_solver(self):
        ''' solve linear FETI problem with PCGP with partial reorthogonalization
        '''

        start_time = time.time()

        logging.info('Assembling  local G, GGT, and e')
        self.assemble_local_G_GGT_and_e()
        build_local_matrix_time = time.time() - start_time
        logging.info('{"elaspsed_time_local_matrix_preprocessing" : %2.4f} # Elapsed time [s]' %(build_local_matrix_time))

        logging.info('Exchange local G_dict and  local e_dict')
        t1 = time.time()
        #G_dict = exchange_global_dict(self.course_problem.G_dict,self.obj_id,self.partitions_list)
        logging.info('{"elaspsed_time_exchange_G_dict" : %2.4f} # Elapsed time [s]' %(time.time() - t1))
        t1 = time.time()
        e_dict = exchange_global_dict(self.course_problem.e_dict,self.obj_id,self.partitions_list)
        logging.info('{"elaspsed_time_exchange_e_dict" : %2.4f} # Elapsed time [s]' %(time.time() - t1))

        #self.course_problem.G_dict = G_dict
        self.course_problem.e_dict = e_dict

        logging.info('Exchange global size')
        t1 = time.time()
        self._exchange_global_size()
        logging.info('{"elaspsed_time_exchange_global_size" : %2.4f} # Elapsed time [s]' %(time.time() - t1))

        t1 = time.time()
        self.assemble_cross_GGT()
        self.GGT_dict = self.course_problem.GGT_dict
        GGT_dict = exchange_global_dict(self.GGT_dict,self.obj_id,self.partitions_list)
        self.course_problem.GGT_dict = GGT_dict
        logging.info('{"elaspsed_time_assemble_GGT_dict" : %2.4f} # Elapsed time [s]' %(time.time() - t1))

        t1 = time.time()
        self.build_local_to_global_mapping()
        logging.info('{"elaspsed_time_build_global_map": %2.4f} # Elapsed time [s]' %(time.time() - t1))

        t1 = time.time()
        GGT = self.assemble_GGT()
        logging.info('{"elaspsed_time_assemble_GGT": %2.4f} # Elapsed time [s]' %(time.time() - t1))

        t1 = time.time()
        G = self.G = G = ParallelRetangularLinearOperator(self.course_problem.G_dict,
        self.local2global_alpha_dofs,self.local2global_lambda_dofs,
        shape=(self.alpha_size , self.lambda_size), neighbors_id = self.neighbors_id)
        logging.info('{"elaspsed_time_parallel_G_operator": %2.4f} # Elapsed time [s]' %(time.time() - t1))

        t1 = time.time()
        e = self.assemble_e()

        logging.info('{"elaspsed_time_assemble_e": %2.4f} # Elapsed time [s]' %(time.time() - t1))

        logging.info('{"primal_variable_size"} = %i' %self.primal_size)
        logging.info('{"dual_variable_size"} = %i'  %self.lambda_size)
        logging.info('{"coarse_variable_size"} = %i' %self.alpha_size)
        
        t1 = time.time()
        lambda_sol,alpha_sol, rk, proj_r_hist, lambda_hist, info_dict = self.solve_dual_interface_problem()
        elaspsed_time_PCPG = time.time() - t1
        logging.info('{"elaspsed_time_PCPG" : %2.4f} # Elapsed time' %(elaspsed_time_PCPG))

        t1 = time.time()
        u_dict, lambda_dict, alpha_dict = self.assemble_solution_dict(lambda_sol,alpha_sol)
        logging.info('{"elaspsed_time_primal_assembly": %2.4f} # Elapsed time [s]' %(time.time() - t1))

        # Serialization the results, Displacement and alpha
        t1 = time.time()
        # serializing displacement
        save_object(u_dict[self.obj_id],'displacement_' + str(self.obj_id) + '.pkl')

        # serializing rigid body correction (alpha)
        try:
            save_object(alpha_dict [self.obj_id],'alpha_' + str(self.obj_id) + '.pkl')
        except:
            pass
        
        elapsed_time = time.time() - start_time
        if self.obj_id == 1:
            sol_obj = Solution({}, lambda_dict, {}, rk, proj_r_hist, lambda_hist, lambda_map=self.local2global_lambda_dofs,
                                alpha_map=self.local2global_alpha_dofs, u_map=self.local2global_primal_dofs,lambda_size=self.lambda_size,
                                alpha_size=self.alpha_size,solver_time=elapsed_time,
                                local_matrix_time = build_local_matrix_time, time_PCPG = elaspsed_time_PCPG, tolerance = self.tolerance,
                                precond = self.precond_type, info_dict = info_dict)

            save_object(sol_obj,'solution.pkl')

        logging.info('{"serialization_time":%2.4f}' %(time.time() - t1))
        logging.info('{"Total_mpisolver_elaspsed_time":%2.4f}' %(time.time() - start_time))
        
    def assemble_local_G_GGT_and_e(self):
        problem_id = self.obj_id
        local_problem = self.local_problem
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
                
    def assemble_cross_GGT(self):
        
       
        logging.debug('Assembling the cross GGT matrix')
        GGT_local_dict = {}
        for key in local_problem.B_local:
            local_id, nei_id = key
            if local_id!=nei_id:
                if local_id not in self.local_alpha_length_dict:
                    Gi = np.array([])
                else:
                    Gi = self.course_problem.G_dict[local_id,nei_id]
                        
                logging.debug(('interface pair = ' + str(local_id) + ',' + str(nei_id)))
                logging.debug(('Gi = ', Gi))
                Gj = exchange_info(Gi,local_id,nei_id)
                logging.debug(('Gj = ', Gj))
                
                try:
                    GGT_local_dict[local_id,nei_id] = Gi.dot(Gj.T)
                except:
                    GGT_local_dict[local_id,nei_id] = np.array([])

            self.course_problem.update_GGT_dict(GGT_local_dict)

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
        local_problem = self.local_problem
        dof_primal_init = 0
        dof_alpha_init = 0
        for local_id in self.partitions_list:
            primal_length = self.local_primal_length_dict[local_id]
            local_primal_dofs = np.arange(primal_length) 
            global_primal_index = dof_primal_init + local_primal_dofs
            dof_primal_init+= primal_length
            self.local2global_primal_dofs[local_id] = global_primal_index
            self.global2local_primal_dofs[tuple(global_primal_index)] = {local_id:local_primal_dofs}

            if local_id in self.local_alpha_length_dict:
                local_alpha_length = self.local_alpha_length_dict[local_id]
                local_alpha_dofs = np.arange(local_alpha_length) 
                global_alpha_index = dof_alpha_init + local_alpha_dofs
                dof_alpha_init+=local_alpha_length
                self.local2global_alpha_dofs[local_id] = global_alpha_index
                self.global2local_alpha_dofs[tuple(global_alpha_index)] = {local_id:local_alpha_dofs}
                    
        dof_lambda_init = 0
        for local_id in self.partitions_list:
            for nei_id in self.partitions_list:
                if nei_id>=local_id:
                    try:
                        local_lambda_length = self.local_lambda_length_dict[local_id,nei_id]
                        local_dofs = np.arange(local_lambda_length) 
                        global_index = dof_lambda_init + local_dofs
                        dof_lambda_init+=local_lambda_length
                        self.local2global_lambda_dofs[local_id,nei_id] = global_index 
                        self.global2local_lambda_dofs[tuple(global_index)] = {(local_id,nei_id):local_dofs}
                    except:
                        pass

        self.lambda_size = dof_lambda_init
        self.alpha_size = dof_alpha_init
        self.primal_size = dof_primal_init
       
    def assemble_G(self):
        try:
            self.G = self.course_problem.assemble_G(self.local2global_alpha_dofs,self.local2global_lambda_dofs,(self.alpha_size ,self.lambda_size))
            return self.G
        except:
            raise('Build local to global mapping before calling this function')
            
    def compute_GGT_inverse(self):
        return self.GGT_inv

    def get_projection(self):
        
        G = self.G 
        GGT_inv = self.GGT_inv
        GT = G.T
        return lambda r : r - GT.dot(GGT_inv.dot(G.dot(r)))

    def compute_lambda_im(self):
        GGT_inv = self.compute_GGT_inverse()
        return  self.G.T.dot((GGT_inv).dot(self.e))

    def apply_F(self, v,  external_force=False, global_exchange=False):
               
        v_dict = self.vector2localdict(v, self.global2local_lambda_dofs)
        gap_dict = self.solve_interface_gap(v_dict,external_force)
        
        #global exchange
        if global_exchange:
            all_gap_dict = exchange_global_dict(gap_dict,self.obj_id,self.partitions_list)
            gap_dict.update(all_gap_dict)

        d = np.zeros(self.lambda_size)
        for interface_id,global_index in self.local2global_lambda_dofs.items():
            try:
                d[global_index] += gap_dict[interface_id]
            except:
                pass

        return -d

    def apply_F_inv(self,v,global_exchange=False,**kwargs):

        # map array to domains dict and then solve force gap
        v_dict = self.vector2localdict(v, self.global2local_lambda_dofs)
        gap_dict = self.solve_interface_force(v_dict,**kwargs)
        
        #global exchange
        if global_exchange:
            all_gap_dict = exchange_global_dict(gap_dict,self.obj_id,self.partitions_list)
            gap_dict.update(all_gap_dict)

        # assemble vector based on dict
        d = np.zeros(self.lambda_size)
        for interface_id,global_index in self.local2global_lambda_dofs.items():
            try:
                d[global_index] += gap_dict[interface_id]
            except:
                pass

        return d
    
    def get_vdot(self):
        ''' This function wraps the parallel dot product
        of to arrays based on PyFETI data structure.
        '''
        return lambda v,w : pardot(v,w,self.obj_id,self.neighbors_id, self.global2local_lambda_dofs,self.partitions_list)

    def solve_interface_gap(self,v_dict=None, external_force=False):
        local_problem = self.local_problem
        u_dict = {}
        ui = local_problem.solve(v_dict,external_force)
        u_dict_local = local_problem.get_interface_dict(ui)
        u_dict.update(u_dict_local)
        for nei_id in local_problem.neighbors_id:
            nei_dict = {}
            nei_dict[nei_id,self.obj_id] = exchange_info(u_dict[self.obj_id,nei_id],self.obj_id,nei_id,isnumpy=True)
            u_dict.update(nei_dict)
        
        # compute gap
        gap_dict = {}
        for interface_id in u_dict:
            local_id, nei_id = interface_id
            if nei_id>local_id:
                gap = u_dict[local_id,nei_id] + u_dict[nei_id,local_id]
                gap_dict[local_id, nei_id] = gap
                gap_dict[nei_id, local_id] = -gap
            
            elif nei_id==local_id:
                logging.warning('Dirichlet contions = 0!')
                gap = u_dict[local_id,nei_id] 
                gap_dict[local_id, nei_id] = gap

        logging.debug(('gap_dict', gap_dict))
        return gap_dict
    
    def solve_interface_force(self,gap_dict,**kwargs):

        local_problem = self.local_problem
        gap_f_dict = {}
        # solving local problem
        f_dict_local = local_problem.apply_schur_complement(gap_dict,**kwargs)
        gap_f_dict.update(f_dict_local)
        
        for nei_id in local_problem.neighbors_id:            
            nei_dict = {}
            nei_dict[nei_id,self.obj_id] = exchange_info(gap_f_dict[self.obj_id,nei_id],self.obj_id,nei_id,isnumpy=True)
            gap_f_dict.update(nei_dict)


        # compute interface force avg
        f_dict = {}
        for interface_id in gap_f_dict:
            local_id, nei_id = interface_id
            if nei_id>local_id:
                gap = gap_f_dict[local_id,nei_id] + gap_f_dict[nei_id,local_id]
                f_dict [local_id, nei_id] = gap
                f_dict [nei_id, local_id] = gap
            elif nei_id==local_id:
                gap = u_dict[local_id,nei_id] 
                f_dict[local_id, nei_id] = gap


        return f_dict
        
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
        local_problem = self.local_problem
        problem_id = self.obj_id
        lambda_dict[problem_id] = self.vector2localdict(lambda_sol, self.global2local_lambda_dofs)
        u_local = local_problem.solve(lambda_dict[problem_id],external_force_bool=True)

        if local_problem.kernel.shape[0]>0:
            alpha_dict.update(self.vector2localdict(alpha_sol, self.global2local_alpha_dofs))
            delta_u = local_problem.rigid_body_correction(alpha_dict[problem_id])
            u_local +=delta_u

        u_dict[problem_id] = u_local
        
        return u_dict, lambda_dict, alpha_dict 



if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    obj_id = rank + 1

    #setting log file
    LOG_FORMAT = "%(levelname)s : %(message)s : Data -> %(asctime)s "
    logging.basicConfig(level=logging.INFO,filename='domain_' + str(obj_id) + '.log', filemode='w', format=LOG_FORMAT)
    
    
    header ='#'*60
    system_argument = sys.argv
    if  len(system_argument)>1:
        mpi_kwargs = {}    
        for arg in system_argument:
            try:
                var, value = arg.split('=')
                try:
                    mpi_kwargs[var] = eval(value)
                except:
                    mpi_kwargs[var] = value
            except:
                logging.debug('Command line argument not understood, arg = %s cannot be splited in variable name + value' %arg)
                pass
        
        looger = logging.getLogger()
        if 'loglevel' in mpi_kwargs:
            new_loglevel = getattr(logging,mpi_kwargs['loglevel'])
            logging.info('Setting a new log level = %i ' %new_loglevel)
            looger.setLevel(new_loglevel)

        
        
        logging.info(header)
        logging.info('MPI rank %i' %rank)
        logging.info('Directory pass to MPI solver = %s' %os.getcwd())
        localtime = localtime = time.asctime( time.localtime(time.time()) )
        start_time = time.time()
        logging.info('Time at start: %s' %localtime)
        logging.info(header)
    

        case_path = mpi_kwargs['prefix'] + str(obj_id) + mpi_kwargs['ext']
        logging.info('Local object name passed to MPI solver = %s' %case_path)
    
        start_time_load = time.time()
        local_problem = load_object(case_path)
        elapsed_time = time.time() - start_time_load
        logging.info('{"load_object": %2.5e} # Elapsed time in seconds' %elapsed_time)
        
        start_time = time.time()
        parsolver = ParallelSolver(obj_id,local_problem,**mpi_kwargs)
        u_i = parsolver.mpi_solver()
        
        comm.Barrier()
        elapsed_time = time.time() - start_time

        logging.info('Total Parallel solver elapsed time after loading data : %f' %elapsed_time)

        localtime = time.asctime( time.localtime(time.time()) )
        logging.info('Time at end: %s' %localtime)
        logging.info(header)
    else:
        print('/n WARNING. No system argument were passed to the MPIsolver. Nothing to do! \n')
        pass
