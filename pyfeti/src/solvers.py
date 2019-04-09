from unittest import TestCase, main
import numpy as np
from scipy import sparse
from pyfeti.src.linalg import ProjectorOperator
import logging

def PCPG(F_action,residual,Projection_action=None,lambda_init=None,
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
            
            lampda_pcpg = lampda_pcpg + alpha_k*pk
            lambda_hist.append(lampda_pcpg)

            rk = rk - alpha_k*Fpk
            
            # set n - 1 data
            yk1 = yk[:]
            pk1 = pk[:]
            wk1 = wk[:]

        if k==(max_int-1):
            logging.info('Maximum iteration was reached, MAX_INT = %i, without converging!' %k)
            logging.info('Projected norm = %2.5e , where the PCPG tolerance is set to %2.5e' %(norm_wk,tolerance))

        return lampda_pcpg, rk, proj_r_hist, lambda_hist


def alpha_calc(yk,wk,pk,Fpk):
    aux1 = yk.T.dot(wk)
    aux2 = pk.T.dot(Fpk)
    
    alpha = float(aux1/aux2)
    
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

if __name__ == '__main__':
    main()