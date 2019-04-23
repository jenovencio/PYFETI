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
from pyfeti.src.feti_solver import CourseProblem, Solution
from pyfeti.src import solvers


from mpi4py import MPI
import os


def exchange_info(local_var,sub_id,nei_id,tag_id=15,isnumpy=False):
    ''' This function exchange info (lists, dicts, arrays, etc) with the 
    neighbors subdomains. Every subdomain has a master objective which receives 
    the info and do some calculations based on it.
    
    Inpus:
        local_var  : python obj
            local object to send and receive
        sub_id: int
            id of the subdomain
        nei_id : int
            neighbor subdomain to send and receive
        isnumpy : Boolean
            sending numpy arrays
    returns
        nei_var : object of the neighbor
    
    '''    

    if isnumpy:
        # sending message to neighbors
        comm.Send(local_var, dest = nei_id-1)
        # receiving messages from neighbors
        var_nei = np.empty(local_var.shape)
        comm.Recv(var_nei,source=nei_id-1)
    else:
        # sending message to neighbors
        tag_num = nei_id + sub_id + tag_id + nei_id*sub_id
        comm.send(local_var, dest = nei_id-1)
        # receiving messages from neighbors
        var_nei = comm.recv(source=nei_id-1)
        
    return var_nei

def exchange_global_dict(local_dict,local_id,partitions_list):
    
    for global_id in partitions_list:
        if global_id!=local_id:
            nei_dict =  exchange_info(local_dict,local_id,global_id)
            if nei_dict:
                local_dict.update(nei_dict)


    return local_dict


class ParallelSolver():
    def __init__(self,obj_id, local_problem, n_int=500, pinv_tolerance=1.0E-8, **kwargs):
        

        self.obj_id = obj_id
        self.local_problem = local_problem

        self.residual = []
        self.lampda_im = []
        self.lampda_ker = []
        
        self.course_problem = CourseProblem(obj_id)
        self.n_int = n_int
        self.pinv_tolerance = pinv_tolerance
        
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
        self.GGT_inv = None
        self.num_partitions = comm.Get_size()
        self.partitions_list = list(range(1,self.num_partitions+1))
        self.tolerance = 1.e-10
        self.n_int = n_int
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
                
        logging.debug(('lambda dict', self.local_lambda_length_dict))
        logging.debug(('alpha dict', self.local_alpha_length_dict))
        logging.debug(('primal dict',self.local_primal_length_dict))
        
    def mpi_solver(self):
        ''' solve linear FETI problem with PCGP with partial reorthogonalization
        '''

        start_time = time.time()
        self.assemble_local_G_GGT_and_e()
        G_dict = exchange_global_dict(self.course_problem.G_dict,self.obj_id,self.partitions_list)
        e_dict = exchange_global_dict(self.course_problem.e_dict,self.obj_id,self.partitions_list)
        self.course_problem.G_dict = G_dict
        self.course_problem.e_dict = e_dict
        
        self._exchange_global_size()

        self.assemble_cross_GGT()
        self.GGT_dict = self.course_problem.GGT_dict
        
        GGT_dict = exchange_global_dict(self.GGT_dict,self.obj_id,self.partitions_list)
        self.course_problem.GGT_dict = GGT_dict
        
        self.build_local_to_global_mapping()
        
        build_local_matrix_time = time.time() - start_time

        
        GGT = self.assemble_GGT()
        logging.info('GGT size = %i' %GGT.shape[0])
        logging.debug(('GGT = ', GGT))
        G = self.assemble_G()
        e = self.assemble_e()
        
        start_time = time.time()
        lambda_sol,alpha_sol, rk, proj_r_hist, lambda_hist = self.solve_dual_interface_problem()
        elaspsed_time_PCPG = time.time() - start_time
        logging.info('{"elaspsed_time_PCPG" : %2.4e} # Elapsed time' %elaspsed_time_PCPG)

        
        u_dict, lambda_dict, alpha_dict = self.assemble_solution_dict(lambda_sol,alpha_sol)
        

        # Serialization the results, Displacement and alpha
        start_time = time.time()
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
                                local_matrix_time = build_local_matrix_time, time_PCPG = elaspsed_time_PCPG)

            save_object(sol_obj,'solution.pkl')

        elapsed_time = time.time() - start_time
        logging.info('{"serialization_time":%2.4e}' %elapsed_time)
        
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
        for local_id,nei_id in self.local_lambda_length_dict:
            if nei_id>local_id:
                local_lambda_length = self.local_lambda_length_dict[local_id,nei_id]
                local_dofs = np.arange(local_lambda_length) 
                global_index = dof_lambda_init + local_dofs
                dof_lambda_init+=local_lambda_length
                self.local2global_lambda_dofs[local_id,nei_id] = global_index 
                self.global2local_lambda_dofs[tuple(global_index)] = {(local_id,nei_id):local_dofs}

        self.lambda_size = dof_lambda_init
        self.alpha_size = dof_alpha_init
        self.primal_size = dof_primal_init
       
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
            
    def assemble_e(self):
        try:
            self.e = self.course_problem.assemble_e(self.local2global_alpha_dofs,self.alpha_size)
            return self.e
        except:
            raise('Build local to global mapping before calling this function')

    def compute_GGT_inverse(self):
        self.GGT_inv = np.linalg.inv(self.GGT)
        return self.GGT_inv

    def compute_lambda_im(self):
        GGT_inv = self.compute_GGT_inverse()
        return  self.G.T.dot((GGT_inv).dot(self.e))

    def apply_F(self, v,  external_force=False):
       
        logging.debug(('global2local_lambda_dofs ', self.global2local_lambda_dofs))
        v_dict = self.vector2localdict(v, self.global2local_lambda_dofs)
        gap_dict = self.solve_interface_gap(v_dict,external_force)
        
        #global exchange
        all_gap_dict = exchange_global_dict(gap_dict,self.obj_id,self.partitions_list)
        comm.Barrier() 
        gap_dict.update(all_gap_dict)

        d = np.zeros(self.lambda_size)
        for interface_id,global_index in self.local2global_lambda_dofs.items():
            d[global_index] += gap_dict[interface_id]

        return -d

    def solve_dual_interface_problem(self,algorithm='PCPG'):
        
        lambda_im = self.compute_lambda_im()
        G = self.G
        GGT_inv = self.GGT_inv
        I = np.eye(self.lambda_size)
        Projection_action = lambda r : (I - G.T.dot(GGT_inv.dot(G))).dot(r)
        F_action = lambda lambda_ker : self.apply_F(lambda_ker)
        residual = -self.apply_F(lambda_im, external_force=True)

        method_to_call = getattr(solvers, algorithm)
        logging.info('Dual Interface algorithm = %s' %algorithm)

        n_int = max(self.lambda_size,self.n_int)
        lambda_ker, rk, proj_r_hist, lambda_hist = method_to_call(F_action,residual,Projection_action=Projection_action,
                                                         lambda_init=None,
                                                         Precondicioner_action=None,
                                                         tolerance=self.tolerance,max_int=n_int)

        lambda_sol = lambda_im + lambda_ker

        alpha_sol = GGT_inv.dot(G.dot(residual - self.apply_F(lambda_ker)))

        return lambda_sol,alpha_sol, rk, proj_r_hist, lambda_hist

    def solve_interface_gap(self,v_dict=None, external_force=False):
        local_problem = self.local_problem
        u_dict = {}
        logging.debug(('v_dict', v_dict))
        ui = local_problem.solve(v_dict,external_force)
        u_dict_local = local_problem.get_interface_dict(ui)
        u_dict.update(u_dict_local)
        for nei_id in local_problem.neighbors_id:
            logging.debug(('nei_id', nei_id))
            logging.debug(('u_dict', u_dict))
            nei_dict = {}
            nei_dict[nei_id,self.obj_id] = exchange_info(u_dict[self.obj_id,nei_id],self.obj_id,nei_id)
            u_dict.update(nei_dict)
        
        logging.debug(('u_dict', u_dict))
        # compute gap
        gap_dict = {}
        for interface_id in u_dict:
            local_id, nei_id = interface_id
            if nei_id>local_id:
                gap = u_dict[local_id,nei_id] + u_dict[nei_id,local_id]
                gap_dict[local_id, nei_id] = gap
                gap_dict[nei_id, local_id] = -gap
        logging.debug(('gap_dict', gap_dict))
        return gap_dict
    
    def vector2localdict(self,v,map_dict):
        v_dict = {}
        for global_index, local_info in map_dict.items():
            for interface_id in local_info:
                v_dict[interface_id] = v[np.ix_(global_index)]

        return v_dict
        
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


def launch_ParallelSolver(rank, tmp_folder='temp' ,  prefix='local_problem_', ext='.pkl',**kwargs):


    obj_id = rank + 1
    case_path = prefix + str(obj_id) + ext

    logging.info('Local problem ID = %s' %obj_id)
    logging.info('Local object name passed to MPI solver = %s' %case_path)

    local_problem = load_object(case_path)
    logging.info(('local_problem obj = ', local_problem))
    parsolver = ParallelSolver(obj_id,local_problem)
    u_i = parsolver.mpi_solver()
    return u_i


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    obj_id = rank + 1
    logging.basicConfig(level=logging.INFO,filename='domain_' + str(obj_id) + '.log')
    
    header ='###################################################################'
    system_argument = sys.argv

    if  len(system_argument)>1:
        mpi_kwargs = {}
        for arg in system_argument:
            try:
                var, value = arg.split('=')
                try:
                    mpi_kwargs[var] = int(value)
                except:
                    mpi_kwargs[var] = value
            except:
                logging.debug('Commnad line argument noy understood, arg = %s cannot be splited in variable name + value' %arg)
                pass


        

        
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
        parsolver = ParallelSolver(obj_id,local_problem)
        u_i = parsolver.mpi_solver()
        
        comm.Barrier()
        elapsed_time = time.time() - start_time

        logging.info('Total Parallel solver elapsed time after loading data : %f' %elapsed_time)

        localtime = time.asctime( time.localtime(time.time()) )
        logging.info('Time at end: %s' %localtime)
        logging.info(header)
    else:
        print('/n WARNING. No system argument were passed to the MPIsolver. Nothing to do! /n')
        pass
