from unittest import TestCase, main
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from pyfeti.src.linalg import ProjectorOperator
import logging
from mpi4py import MPI
import time

def PCPG(F_action,residual,Projection_action=None,lambda_init=None,
        Precondicioner_action=None,tolerance=None,max_int=None,
        callback=None,vdot= None,save_lambda=False,exact_norm=True):
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

        tolerance: float, Default= None
            convergence tolerance, if None tolerance=1.e-10

        max_int : int, Default= None
            maximum number of iterations, if None max_int = int(1.2*len(residual))

        callback : callable, Default None
            function to be callabe at the and of each iteration

        vdot : callable, Default None
            function with the dot product of vdot(v,w) if none 
            then, np.dot(v,w)

        save_lambda : Booelan, Default = False
            store lambda interations in the a list

        exact_norm : Booelan, Default = False
            if True compute the L2 norm of the projected residual sqrt(vdot(wk,wk)), if false compute 
            the sqrt(vdot(wk,yk)) where yk is the projected preconditioned array

        return 
            lampda_pcgp : np.array
                last lambda
            rk : np.array
                last projected residual
            proj_r_hist : list
                list of the history of the norm of the projected residuals
            lambda_hist : list
            list of the 

        '''
         
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        interface_size = len(residual)
        apply_precond = True
        Identity = LinearOperator(dtype=residual.dtype,shape=(interface_size,interface_size), matvec = lambda x : x)
         
        if tolerance is None:
            tolerance=1.e-10

        if max_int is None:
            max_int = int(1.2*interface_size)

        if lambda_init is None:
            lampda_pcpg = np.zeros(interface_size)
        else:
            lampda_pcpg = lambda_init
                     
        if Precondicioner_action is None:
            #Precond = np.eye(interface_size,interface_size).dot
            Precond = Identity.dot
            apply_precond = False
        else:
            Precond = Precondicioner_action

        if Projection_action is None:
            #P = np.eye(interface_size,interface_size).dot
            P = Identity.dot
        else:
            P = Projection_action
            
        if vdot is None:
            vdot = lambda v,w : np.dot(v,w)

        # defining a norm based on vdot function
        norm_func =  lambda v : np.sqrt(vdot(v,v))

        F = F_action

        logging.info('Setting PCPG tolerance = %4.2e' %tolerance)
        logging.info('Setting PCPG max number of iterations = %i' %max_int)

        # initialize variables
        info_dict = {}
        global_start_time = time.time()
        beta = 0.0
        yk1 = np.zeros(interface_size)
        wk1 = np.zeros(interface_size)
        proj_r_hist = []
        lambda_hist = []
        rk = residual
        k=0
        for k in range(max_int):
            
            logging.info('#'*60)
            logging.info('PCPG Iteration = %i' %(k))
            info_dict[k] = {}

            proj_start = time.time()
            wk = P(rk)  # projection action
            proj_elapsed_time = time.time() - proj_start
            logging.info('{"elaspsed_time_projection" : %2.4f} # Elapsed time' %(proj_elapsed_time))
            info_dict[k]["elaspsed_time_projection"] = proj_elapsed_time
            # checking if precond will be applied, if not extra projection must be avoided
            t1 = time.time()
            if apply_precond:
                zk = Precond(wk)
                yk = P(zk)
            else:
                zk = wk
                yk = zk
            info_dict[k]["elaspsed_time_precond"] = time.time() - t1
            
            beta_start = time.time()
            if k>1:
                vn = vdot(yk,wk)
                beta = vn/vn1
                vn1 = vn
            else:
                vn1 = vdot(yk,wk)
                vn = vn1 
                pk1 = yk
            beta_elapsed_time = time.time() - beta_start
            logging.info('{"elaspsed_time_beta" : %2.4f} # Elapsed time' %(beta_elapsed_time))
            info_dict[k]["elaspsed_time_beta"] = beta_elapsed_time

            if exact_norm:
                norm_wk = norm_func(wk)
                logging.info('Iteration = %i, Norm of project residual wk = %2.5e.' %(k,norm_wk))
                if norm_wk<=tolerance:
                    logging.info('PCG has converged after %i' %(k+1))
                    break
            else:
                norm_wk = np.sqrt(vn1)
                logging.info('Iteration = %i, Norm of project preconditioned residual  sqrt(<yk,wk>) = %2.5e!' %(k,norm_wk))
                if norm_wk<=tolerance:
                    #evaluate the exact norm
                    _norm_wk = norm_func(wk)
                    if _norm_wk<=tolerance:
                        logging.info('PCG has converged after %i' %(k+1))
                        logging.info('Iteration = %i, Norm of project residual wk = %2.5e!' %(k,_norm_wk))
                        break

            proj_r_hist.append(norm_wk)
            
            pk = yk + beta*pk1

            F_start = time.time()
            Fpk = F(pk)
            F_elapsed_time = time.time() - F_start
            logging.info('{"elaspsed_time_F_action" : %2.4f} # Elapsed time' %(F_elapsed_time))
            info_dict[k]["elaspsed_time_F_action"] = F_elapsed_time

            alpha_start = time.time()
            alpha_k = alpha_calc(vn1,pk,Fpk,vdot)
            alpha_elapsed_time = time.time() - alpha_start
            logging.info('{"elaspsed_time_alpha" : %2.4f} # Elapsed time'  %(alpha_elapsed_time))
            info_dict[k]["elaspsed_time_alpha"] = alpha_elapsed_time

            lampda_pcpg = lampda_pcpg + alpha_k*pk
            
            if save_lambda:
                lambda_hist.append(lampda_pcpg)

            rk = rk - alpha_k*Fpk
            
            # set n - 1 data
            yk1 = yk[:]
            pk1 = pk[:]
            wk1 = wk[:]

            if callback is not None:
                callback(lampda_pcpg)

            elapsed_time = time.time() - proj_start
            
            logging.info('{"elaspsed_time_PCPG_iteration" : %2.2f} # Elapsed time' %(elapsed_time))
            info_dict[k]["elaspsed_time_iteration"] = elapsed_time

        if (k>0) and k==(max_int-1):
            logging.warning('Maximum iteration was reached, MAX_INT = %i, without converging!' %(k+1))
            logging.warning('Projected norm = %2.5e , where the PCPG tolerance is set to %2.5e' %(norm_wk,tolerance))

        elapsed_time = time.time() - global_start_time
        logging.info('#'*60)
        logging.info('{"Total_elaspsed_time_PCPG" : %2.2f} # Elapsed time [s]' %(elapsed_time))
        logging.info('Number of PCPG Iterations = %i !' %(k+1))
        avg_iteration_time = elapsed_time/(k+1)
        logging.info('{"avg_iteration_time_PCPG" : %2.4f} # Elapsed time [s]' %(avg_iteration_time))
        logging.info('#'*60)

        info_dict['avg_iteration_time'] = elapsed_time/(k+1)
        info_dict['Total_elaspsed_time_PCPG'] = elapsed_time
        info_dict['PCPG_iterations'] = k+1
        return lampda_pcpg, rk, proj_r_hist, lambda_hist, info_dict


def alpha_calc(vn1,pk,Fpk,vdot=None):
    if vdot is None:
        vdot = lambda v,w : np.dot(v,w)

    aux2 = vdot(pk,Fpk)
    alpha = vn1/aux2
    return alpha

def pminres(F_action,residual,Projection_action=None,lambda_init=None,
        Precondicioner_action=None,tolerance=1.e-10,max_int=500):
        ''' This function is a general interface for Scipy MinRes algorithm

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
            lampda_pcpg = np.zeros(interface_size)
        else:
            lampda_pcpg = lambda_init
         
        if Precondicioner_action is None:
            Precond = np.eye(interface_size,interface_size).dot
        else:
            Precond = Precondicioner_action

        if Projection_action is None:
            P = np.eye(interface_size,interface_size).dot
        else:
            P = Projection_action
            
        #F = F_action
        F = sparse.linalg.LinearOperator((interface_size,interface_size), matvec=F_action)
        P = sparse.linalg.LinearOperator((interface_size,interface_size), matvec=P, rmatvec = P)

        PF= ProjectorOperator(F,P,shape=(interface_size,interface_size))
        b = residual - F.dot(lampda_pcpg)
        #lampda_pcpg, info = sparse.linalg.minres(PF,b,x0=lampda_pcpg,tol=tolerance,maxiter=max_int,show=False,check=False)
        lampda_pcpg, info = sparse.linalg.minres(F,b,x0=lampda_pcpg,M=P,tol=tolerance,maxiter=max_int,show=False,check=False)

        rk, proj_r_hist, lambda_hist = PF.dot(lampda_pcpg) - b, None, None

        if  info==0:
            logging.info('Project MinRes has converged')
        else:
            logging.info('Project MinRes has NOT converged after %i iterations' %info)

        return lampda_pcpg, rk , proj_r_hist, lambda_hist


class  Test_solvers(TestCase):

    def test_ProjectorOperator_with_minres(self):
        A = 3*np.array([[2,-1,0],[-1,2,0],[0,-1,2]])
        P = np.array([[1,0,0],[0,1,0],[0,0,0]])
        b = np.array([-2,4,0])
        Asingular = P.dot(A.dot(P))

        F_action = lambda x : A.dot(x)
        Projection_action = lambda x : P.dot(x)
        residual = b

        x_minres, rk, proj_r_hist, lambda_hist = pminres(F_action,residual,Projection_action)
        x_svd = (np.linalg.pinv(Asingular)).dot(b)
        np.testing.assert_almost_equal(x_minres,x_svd,decimal=10)

    def test_PCPG(self):
        A = 3*np.array([[2,-1,0],[-1,2,-1],[0,-1,1]])
        b = np.array([-2,4,0])

        x_target = np.linalg.solve(A,b)
        r = b
        x_pcpg, rk , proj_r_hist, X_hist = PCPG(A.dot,r,max_int=6)
        np.testing.assert_array_almost_equal(x_target,x_pcpg,decimal=10)


if __name__ == '__main__':
    main()