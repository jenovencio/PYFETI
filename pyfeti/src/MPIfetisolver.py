# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 08:35:26 2017

@author: ge72tih
"""

# importing mpy4py
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import subprocess
import sys
import dill as pickle
import amfe
import numpy as np
import scipy.sparse as sparse
import scipy
import logging
import pandas as pd

#logging.basicConfig(level=logging.INFO)

def run_command(cmd):
    """given shell command, returns communication tuple of stdout and stderr"""
    return subprocess.Popen(cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            stdin=subprocess.PIPE).communicate()

                            
                            
# Subdomain level
def subdomain_i(sub_id):    
    submesh_i = domain.groups[sub_id]
    sub_i = amfe.FETIsubdomain(submesh_i)
    B_dict = sub_i.assemble_interface_boolean_matrix()
    sub_i.calc_null_space()
    R = sub_i.null_space

    # store all G in G_dict
    Gi_dict = {}
    for key in B_dict:
        if sub_i.zero_pivot_indexes:
            Gi_dict[key] = -B_dict[key].dot(R)
        else:
            Gi_dict[key] = None
        
    return sub_i, Gi_dict


def create_GtG_rows(Gi_dict,Gj_dict,sub_id):
        

    GtG_dict = {}

    for local_id, nei_id in Gi_dict:
        if local_id == sub_id:
                                    
            try:
                Gi = Gi_dict[sub_id,nei_id]
                GiGi = Gi.T.dot(Gi)
                GtG_dict[sub_id,sub_id] = GiGi
                
                Gj = Gj_dict[nei_id,sub_id]
                GiGj = Gi.T.dot(Gj)
                GtG_dict[sub_id,nei_id] = GiGj
                
            except:
                pass
             
            
    return GtG_dict 


def exchange_info(sub_id,master,master_append_func,var,partitions_list):
    ''' This function exchange info (lists, dicts, arrays, etc) with the 
    neighbors subdomains. Every subdomain has a master objective which receives 
    the info and do some calculations based on it.
    
    Inpus:
        sub_id: id of the subdomain
        master: object with global information
        master_append_func: master function to append the exchanged var
        var: variable to be exchanged among neighbors
        partitions_list: list of subdomain neighbors
    
    '''
    
    # forcing list as inputs
    if type(var)!=list:
        var_list = [var]
    else:        
        var_list = var
        
    if type(master_append_func)!=list:
        master_append_func_list = [master_append_func]
    else:
        master_append_func_list = master_append_func
    
    if len(var)!=len(master_append_func):
        logging.warning('Error exchanging information among subdomains')
        return None
    
    for var,master_append_func in zip(var_list,master_append_func_list):
        for partition_id in partitions_list:
            if partition_id != sub_id:
                #send to neighbors local h 
                comm.send(var, dest=partition_id)
                # receive local h from neighbors
                nei_var = comm.recv(source=partition_id)

                try:
                    master_append_func(nei_var)
             
                except:       
                    master_append_func(nei_var,partition_id)
            else:
                try:   
                    print(var)
                    master_append_func(var)
             
                except:
                    master_append_func(var,sub_id)
    
    return master


def subdomain_apply_F(sub_i,lambda_id_dict,pk):
    ''' This step is to calculate a_hat in every domain
    '''
        
    i = 0
    sub_id = sub_i.submesh.key
    for nei_id in sub_i.submesh.neighbor_partitions:
        local_id = master.lambda_id_dict[sub_id,nei_id]
        Bi = sub_i.B_dict[sub_id,nei_id].todense()    
        local_pk = pk[local_id]
        
        b_hati = np.matmul(Bi.T,local_pk)
        if i == 0:
            b_hat = b_hati
            i = 1
        else:
            b_hat = b_hat + b_hati
            
    # solving K(s)a(s) = b(s) in order to calculte the action of F(s)
    Ui = sub_i.full_rank_upper_cholesky.todense()
    idf = sub_i.zero_pivot_indexes
    b_hat[idf] = 0.0
    
    a_hat = scipy.linalg.cho_solve((Ui,False),b_hat)  
    

    # build local h with local F(v)    
    local_h_dict = {}    
    for nei_id in sub_i.submesh.neighbor_partitions:
        Bi = sub_i.B_dict[sub_id,nei_id].todense()    
        local_h_dict[sub_id,nei_id] = Bi.dot(a_hat)
       
    # sending local h for master 
    return local_h_dict
        

def global_apply_F(master,sub_i,lambda_id_dict,pk):
    
    sub_id = sub_i.submesh.key
    local_h_dict = subdomain_apply_F(sub_i,lambda_id_dict,pk)
    
    
    master_func_list = [master.append_h,master.append_d_hat]
    var_list = [local_h_dict,sub_i.dual_force_dict]
    
    master = exchange_info(sub_id,master,master_func_list,var_list,partitions_list)
    
        
    return master.assemble_h()


def subdomain_step4(sub_i,lambda_sol,alpha):
            
    i = 0
    for nei_id in sub_i.submesh.neighbor_partitions:
        local_id = master.lambda_id_dict[sub_id,nei_id]
        Bi = sub_i.B_dict[sub_id,nei_id].todense()    
        local_lambda = lambda_sol[local_id]
        
        if (sub_id,nei_id) in master.alpha_dict:
            alpha_id = master.alpha_dict[sub_id,nei_id]
        
        
        b_hati = np.matmul(Bi.T,local_lambda)
        if i == 0:
            b_hat = b_hati
            i = 1
        else:
            b_hat = b_hat + b_hati


    f = sub_i.force
    b = f - b_hat
    
    
    # solving K(s)a(s) = b(s) in order to calculte the action of F(s)
    Ui = sub_i.full_rank_upper_cholesky.todense()
    idf = sub_i.zero_pivot_indexes
    b[idf] = 0.0
    u_hat = scipy.linalg.cho_solve((Ui,False),b)  
    
    if idf:
        R = sub_i.null_space
        local_alpha = alpha[alpha_id]
        u_bar = u_hat + np.matmul(R,local_alpha)
        sub_i.displacement = u_bar
    else:    
        sub_i.displacement = u_hat
            
    return sub_i.displacement


def beta_calc(k,y_dict=None,w_dict=None):
    
    if k == 0:
        return 0.0
    
    else:
        yk1 = y_dict[0]
        yk2 = y_dict[1]
        wk1 = w_dict[0]
        wk2 = w_dict[1]
        
        aux1 = float(yk1.T.dot(wk1))
        aux2 = float(yk2.T.dot(wk2))
        
        beta = aux1/aux2
        
        return beta
        
        
def alpha_calc(yk,wk,pk,Fpk):
    aux1 = yk.T.dot(wk)
    aux2 = pk.T.dot(Fpk)
    
    alpha = float(aux1/aux2)
    
    return alpha


    
def action_of_global_F_mpi(global_lambda,sub_i,master,partitions_list):
    ''' This function apply the action of F in lambda
    communacation among subdomain using MPI
    '''
    #%%%%%%%%%%%%%%% APPLYING LOCAL F  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # apply local F operations
    local_h_dict = sub_i.apply_local_F(global_lambda, master.lambda_dict)
    master.append_h(local_h_dict)
    #%%%%%%%%%%%%%%%%%%%%%% END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #%%%%%%%%%%% EXCHANGE INFORMATION WITH ALL SUBDOMAIN  %%%%%%%%%%%%
    master_func_list = [master.append_h]
    var_list = [local_h_dict]
    exchange_info(sub_i.key,master,master_func_list,var_list,partitions_list)
    #%%%%%%%%%%%%%%%%%%%%%% END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #%%%%%%%%%%%%% ASSEMBLE GLOBAL F*Lambda  %%%%%%%%%%%%%%%%%%%
    Fim = master.assemble_global_F_action() # Fpk = B*Kpinv*B'*pk
    logging.debug('Fim')
    logging.debug(Fim)
    return Fim
    
def assemble_global_d_mpi(sub_i,master,partitions_list):        
    #%%%%%%%%%%% EXCHANGE INFORMATION WITH ALL SUBDOMAIN  %%%%%%%%%%%%
    dual_force_dict = sub_i.calc_dual_force_dict()
    master_func_list = [master.append_d_hat]
    var_list = [dual_force_dict]
    exchange_info(sub_i.key,master,master_func_list,var_list,partitions_list)
    #%%%%%%%%%%%%%%%%%%%%%% END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    d = master.assemble_global_d_hat()
    logging.debug('d')
    logging.debug(d)
    return d
    
def projection_action_mpi(rk,master):
    ''' Project action on rk
    
    '''
    wk, alpha_hat = master.solve_corse_grid(rk)
    return wk
    
def PCGP(F_action,residual,Projection_action=None,lambda_init=None,
        Precondicioner_action=None,tolerance=1.e-10,max_int=500):
        ''' This function is a general interface for PCGP algorithms

        argument:
        F_action: callable function
        callable function that acts in lambda

        residual: np.array
        array with initial interface gap

        lambda_init : np.array
        intial lambda array

        Projection_action: callable function
        callable function to project the residual

        Precondicioner_action : callable function
        callable function to atcs as a preconditioner operator
        in the array w

        tolerance: float
        convergence tolerance

        max_int = int
        maximum number of iterations

        return 
        lampda_pcgp : np.array
            last lambda
        rk : np.array
            last projected residual
        proj_r_hist : list
            list of the history of the norm of the projected  residuals
        lambda_hist : list
            list of the 

        '''
         
        interface_size = len(residual)
         
        if lambda_init is None:
            lampda_pcgp = np.zeros(interface_size)
        else:
            lampda_pcgp = lambda_init
         
        if Precondicioner_action is None:
            Precond = np.eye(interface_size,interface_size).dot
        else:
            Precond = Precondicioner_action

        if Projection_action is None:
            P = np.eye(interface_size,interface_size).dot
        else:
            P = Projection_action
            
        F = F_action

        # initialize variables
        beta = 0.0
        yk1 = np.zeros(interface_size)
        wk1 = np.zeros(interface_size)
        proj_r_hist = []
        lambda_hist = []
        rk = residual
        for k in range(max_int):
            wk = P(rk)  # projection action
            
            norm_wk = np.linalg.norm(wk)
            proj_r_hist.append(norm_wk)
            if norm_wk<tolerance:
                logging.info('PCG has converged after %i' %(k+1))
                break

            zk = Precond(wk)
            yk = P(zk)

            if k>1:
                beta = yk.T.dot(wk)/yk1.T.dot(wk1)
            else:
                pk1 = yk

            pk = yk + beta*pk1
            Fpk = F(pk)
            alpha_k = alpha_calc(yk,wk,pk,Fpk)
            
            lampda_pcgp = lampda_pcgp + alpha_k*pk
            lambda_hist.append(lampda_pcgp)

            rk = rk - alpha_k*Fpk
            
            # set n - 1 data
            yk1 = yk[:]
            pk1 = pk[:]
            wk1 = wk[:]

        return lampda_pcgp, rk, proj_r_hist, lambda_hist
        
    
class ParallelSolver():
    def __init__(self):
        
        self.residual = []
        self.lampda_im = []
        self.lampda_ker = []
        
    def mpi_solver(self,sub_domain,num_partitions, n_int=500, cholesky_tolerance=1.0E-8):
        ''' solve linear FETI problem with PCGP with parcial reorthogonalization
        '''
        
        if rank<=num_partitions and rank>0:
            global sub_id
            sub_id = rank
            logging.info("%%%%%%%%%%%%%%%%%%% START %%%%%%%%%%%%%%%%%%%%%%%%%%%%")    
            logging.info("Solving domain %i from size %i" %(rank,num_partitions))   
            sub_i = amfe.FETIsubdomain(sub_domain.groups[sub_id])
            sub_i.set_cholesky_tolerance =  cholesky_tolerance
            Gi_dict = sub_i.calc_G_dict()
            
            # append local dofs info in master to build global indexation matrices
            subdomain_interface_dofs_dict = sub_i.num_of_interface_dof_dict
            subdomain_null_space_size = sub_i.null_space_size
            local_info_dict =  master.append_partition_dof_info_dicts(sub_id,
                                                                      subdomain_interface_dofs_dict,
                                                                      subdomain_null_space_size)

            #%%%%%%%%%% START SENDING AND RECEIVING G MATRIX %%%%%%%%%%%%%%%%%%%
            # sending G_dict for neighbors
            for nei_id in sub_i.submesh.neighbor_partitions:
                logging.info("\nSending message from %i to neighbor %i" %(sub_id,nei_id))
                if (sub_id,nei_id) in Gi_dict: 
                    comm.send(Gi_dict[sub_id, nei_id], dest=nei_id)
            
            # receiving message for neighbors
            Gj_dict = {}
            for nei_id in sub_i.submesh.neighbor_partitions:
                logging.debug("\nReceiving message at subdomain %i from neighbor %i" %(sub_id,nei_id))
                Gj_dict[nei_id,sub_id]= comm.recv(source=nei_id)
            
            #%%%%%%%%%% END ENDING AND RECEIVING G MATRIX %%%%%%%%%%%%%%%%%%%%
            
            #%%%%%%%%%% START APPENDING G MATRIX IN MASTER %%%%%%%%%%%%%%%%%%%
            # append local G_dict and neighbors G_dict
            master.append_G_dict(Gi_dict)
            master.append_G_dict(Gj_dict)
                     
            logging.debug('G_dict')
            logging.debug(master.G_dict)
            
            #%%%%%%%%%% END APPENDING G MATRIX IN MASTER %%%%%%%%%%%%%%%%%%%%%
            
            
            #%%%%%%%%%% COMPUTING GtG_rows  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # append neighbor G_dict to subdomain
            for nei_key, sub_key in Gj_dict:
                Gj = Gj_dict[nei_id,sub_id]
                sub_i.append_neighbor_G_dict(nei_key,Gj)
            
            # creating GtG rows
            GtG_rows_dict = sub_i.calc_GtG_row()
            null_space_size = sub_i.null_space_size
            
            logging.debug('Local GtG rows')
            logging.debug(GtG_rows_dict.keys())
                    
            #%%%%%%%%%%%%%%% END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
            #%%%%%%%%%%% EXCHANGE INFORMATION WITH ALL SUBDOMAIN  %%%%%%%%%%%%
            # List of function in master to append variables
            master_func_list = [master.appendGtG_row, 
                                master.append_G_dict,
                                master.append_null_space_force,
                                master.append_partition_tuple_info]
            
            # variables to be appended in master
            var_list = [GtG_rows_dict,
                        master.G_dict,
                        sub_i.null_space_force,
                        (sub_id,subdomain_interface_dofs_dict,subdomain_null_space_size)]
            
            # exchange information among all partitions
            exchange_info(sub_id,master,master_func_list,var_list,partitions_list)
            
            logging.debug('Master GtG keys')
            logging.debug(master.GtG_row_dict.keys())    
            
            master.build_local_to_global_mapping()
            logging.info('total interface dof %i' %master.total_interface_dof)
            logging.info('Null space size %i' %master.total_nullspace_dof)
            #%%%%%%%%%%%%%%%%%%%%%% END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            #%%%%%%%%%%%%%%% SOLVING SELF EQUILIBRIUM LAMBDA IM  %%%%%%%%%%%%%%
            lambda_im = master.solve_lambda_im()
            lambda_ker = master.lambda_ker
            #%%%%%%%%%%%%%%%%%%%%%% END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            # defining solver actions
            F = lambda x : action_of_global_F_mpi(x,sub_i,master,partitions_list)
            P = lambda rk : projection_action_mpi(rk,master)
            
            #%%%%%%%%%%%%%%% APPLYING LOCAL F  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Fim = F(lambda_im)
            d = assemble_global_d_mpi(sub_i,master,partitions_list)
            # initial residual
            r0 = d - Fim
            #---------------------------------------------------------------------
            # PCPG algorithm     
            lambda_ker, last_res, proj_r_hist, lambda_hist = amfe.PCGP(F,r0,P)
                
            # lagrange multiplier solution
            lambda_sol =  lambda_im + lambda_ker

            # compute global error without Rigid body correction
            d = master.assemble_global_d_hat() # dual force global assemble
            
            # calc Global F_lambda
            d_hat = d - F(lambda_sol)
        
            # calc Rigid Body Correction
            wk, global_alpha = master.solve_corse_grid(d_hat)
            
            sub_i.solve_local_displacement(global_lambda=lambda_sol, lambda_dict=master.lambda_dict)
            sub_i.apply_rigid_body_correction(global_alpha,master.alpha_dict)
            
            res_path = os.path.join(directory,str(sub_id) + '.are')
            amfe.save_object(sub_i, res_path)
            return sub_i
    
            logging.info("%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%")    
            
        else:
            logging.info("Nothing to do on from process %i " %rank)   
            return None


if __name__ == "__main__":
    # execute only if run as a script
    domain_pkl_path = sys.argv        

    args = []
    for s in sys.argv:
        args.append(s)    
        
    # load FEA case
    case_obj = args[1]
    try:
        directory = args[2]
    except:
        directory = ''
    case_path = os.path.join(directory,case_obj)
    
    logging.debug('########################################')
    logging.debug('Case object = %s' %case_obj)
    logging.debug('Directory pass to MPI solver = %s' %directory)
    logging.debug('FUll case path passed to MPI solver = %s' %case_path)
    logging.debug('########################################')
    
    my_system = amfe.load_obj(case_path)
    domain = my_system.domain
    # Instanciating Global Master class to handle Coarse problem
    
    
    num_partitions = len(domain.groups)
    partitions_list = np.arange(1,num_partitions+1)
    
    # instantiating master to handle global information
    master = amfe.Master()
    master.subdomain_keys = partitions_list
    
    # instantiating parallel solver
    parsolver = ParallelSolver()
    sub_i = parsolver.mpi_solver(domain,num_partitions)
    
    if rank == 1:
        solver_path = os.path.join(directory, 'solver.sol')
        amfe.save_object(parsolver, solver_path)




