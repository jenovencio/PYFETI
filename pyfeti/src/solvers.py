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
        num_righthand_sides = residual.ndim

        apply_precond = True
        Identity = LinearOperator(dtype=residual.dtype,shape=(interface_size,interface_size), matvec = lambda x : x)
         
        if tolerance is None:
            tolerance=1.e-10

        if max_int is None:
            max_int = int(1.2*interface_size)

        if lambda_init is None:
            lampda_pcpg = np.zeros(shape=residual.shape)
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
        norm_func =  lambda v : np.sqrt(vdot(v.conj(),v)).real

        F = F_action

        logging.info('Setting PCPG tolerance = %4.2e' %tolerance)
        logging.info('Setting PCPG max number of iterations = %i' %max_int)

        # initialize variables
        info_dict = {}
        global_start_time = time.time()
        beta = np.zeros(shape=(num_righthand_sides,))
        yk1 = np.zeros(shape=residual.shape)
        wk1 = np.zeros(shape=residual.shape)
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
                vn = vdot(yk.conj(),wk)
                beta = vn/vn1
                vn1 = vn
            else:
                vn1 = vdot(yk.conj(),wk)
                vn = vn1 
                pk1 = yk
            beta_elapsed_time = time.time() - beta_start
            logging.info('{"elaspsed_time_beta" : %2.4f} # Elapsed time' %(beta_elapsed_time))
            info_dict[k]["elaspsed_time_beta"] = beta_elapsed_time

            if exact_norm:
                norm_wk = norm_func(wk).max()
                logging.info('Iteration = %i, Norm of project residual wk = %2.5e.' %(k,norm_wk))
                if norm_wk<=tolerance:
                    logging.info('PCG has converged after %i' %(k+1))
                    break
            else:
                norm_wk = np.sqrt(vn1).max()
                logging.info('Iteration = %i, Norm of project preconditioned residual  sqrt(<yk,wk>) = %2.5e!' %(k,norm_wk))
                if norm_wk<=tolerance:
                    #evaluate the exact norm
                    _norm_wk = norm_func(wk)
                    if _norm_wk<=tolerance:
                        logging.info('PCG has converged after %i' %(k+1))
                        logging.info('Iteration = %i, Norm of project residual wk = %2.5e!' %(k,_norm_wk))
                        break

            proj_r_hist.append(norm_wk)
            
            pk = yk + np.multiply(beta,pk1)

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

            lampda_pcpg = lampda_pcpg + np.multiply(alpha_k,pk)
            
            if save_lambda:
                lambda_hist.append(lampda_pcpg)

            rk = rk - np.multiply(alpha_k,Fpk)
            
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

    aux2 = vdot(pk.conj(),Fpk)
    alpha = vn1/aux2
    return alpha


def PGMRES(F_action,residual,Projection_action=None,lambda_init=None,
        Precondicioner_action=None,tolerance=None,max_int=None,
        callback=None,vdot= None,save_lambda=False,exact_norm=True,restart=None,
        atol=1.e-12):
        ''' This function is a general interface for PGMRES algorithms

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

        restart : int, default = None
            number of iteratios before restart

        atol : float
            tolerance for the entries of the Hesenberg matrix

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
        num_righthand_sides = residual.ndim

        apply_precond = True
        Identity = LinearOperator(dtype=residual.dtype,shape=(interface_size,interface_size), matvec = lambda x : x)
         
        if tolerance is None:
            tolerance=1.e-10

        if max_int is None:
            max_int = int(1.2*interface_size)

        if lambda_init is None:
            lampda_pcpg = np.zeros(shape=residual.shape)
            lambda_init = np.zeros(shape=residual.shape)
        else:
            lampda_pcpg = lambda_init
                     
        if Precondicioner_action is None:
            Precond = Identity.dot
            apply_precond = False
        else:
            Precond = Precondicioner_action

        if Projection_action is None:
            P = Identity.dot
        else:
            P = Projection_action
            
        if vdot is None:
            vdot = lambda v,w : np.dot(v,w)

        # defining a norm based on vdot function
        norm_func =  lambda v : np.sqrt(vdot(v.conj(),v)).real

        F = F_action

        logging.info('Setting PGMRES tolerance = %4.2e' %tolerance)
        logging.info('Setting PGMRES max number of iterations = %i' %max_int)


        M = LinearOperator(dtype=residual.dtype,shape=(interface_size,interface_size), matvec = lambda x : P(Precond(P(x))))
        A = LinearOperator(dtype=residual.dtype,shape=(interface_size,interface_size), matvec = lambda x : F(x))

        if restart is None:
            restart = int(min(0.1*max_int,20,2*A.shape[0])) 

        # initialize variables
        info_dict = {}
        proj_r_hist = []
        lambda_hist = []
        rk = M.dot(residual)
        r_norm_0, q, h = PGMRES_initialization(A,rk,restart,vdot=vdot)
        k=0
        for ki in range(max_int):
            y = (M.dot(A.dot(q[:,k]))).reshape(-1)
            for j in range(k+1):
                h[j, k] = vdot(q[:,j], y.conj())
                y = y - h[j, k] * q[:,j]
            h[k + 1, k] = norm_func(y)
            if (np.abs(h[k + 1, k]) <= atol) or k==restart-1:
                k = restart
            else:
                q[:,k + 1] = (y / h[k + 1, k])
                k+=1
                continue
                
                
            if k==restart:
                e1 = np.zeros(k+1,dtype=residual.dtype)  
                e1[0] = 1.0
                result = np.linalg.lstsq(h[:k+1,:k+1], r_norm_0*e1,rcond=-1)[0]
                lampda_pcpg =  lambda_init + q[:,:k+1].dot(result)
                if save_lambda:
                    lambda_hist.append(lampda_pcpg)

                rk =  residual - A.dot(lampda_pcpg) 
                r_norm_0, q, h = PGMRES_initialization(A,rk,restart,vdot=vdot)

                proj_r_hist.append(r_norm_0)

                if r_norm_0 <= max(tolerance,tolerance*r_norm_0):
                    break
                lambda_init=lampda_pcpg
                k=0

        return lampda_pcpg, rk, proj_r_hist, lambda_hist, info_dict

def PGMRES_initialization(A,r,nmax_iter,vdot=None):
    
    if vdot is None:
        vdot = lambda v,w : np.dot(v,w)

    q = np.zeros((A.shape[0],nmax_iter),dtype=A.dtype)
    r_norm = np.sqrt(vdot(r.conj(),r))
    q[:,0] = r / r_norm
    h = np.zeros((nmax_iter + 1, nmax_iter),dtype=A.dtype)
    return r_norm, q, h

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


def GMRes(A, b, x0=None, tol=1.0E-6, atol=1.0E-12, nmax_iter=6, restart=None):
    
    if x0 is None:
        x0 = np.zeros(shape=b.shape, dtype=b.dtype)

    if restart is None:
        restart = int(min(0.1*nmax_iter,20,2*A.shape[0])) 

    r, r_norm_0, q, h = init_GMres(A,b,x0,restart)
    k=0
    for ki in range(nmax_iter):
        y = (A.dot(q[:,k])).reshape(-1)
        for j in range(k+1):
            h[j, k] = np.dot(q[:,j], y.conj())
            y = y - h[j, k] * q[:,j]
        h[k + 1, k] = np.linalg.norm(y)
        if (np.abs(h[k + 1, k]) <= atol) or k==restart-1:
            k = restart
        else:
            q[:,k + 1] = y / h[k + 1, k]
            k+=1
            continue
            
            
        if k==restart:
            e1 = np.zeros(k+1,dtype=b.dtype)  
            e1[0] = 1.0
            result = np.linalg.lstsq(h[:k+1,:k+1], r_norm_0*e1,rcond=-1)[0]
            xk =  x0 + q[:,:k+1].dot(result)

            r, r_norm_0, q, h = init_GMres(A,b,xk,restart)
            if r_norm_0 <= max(tol,tol*r_norm_0):
                break
            x0=xk
            k=0

    return xk

def init_GMres(A,b,x0,nmax_iter):
    r = b - A.dot(x0)
    q = np.zeros((A.shape[0],nmax_iter),dtype=b.dtype)
    r_norm = np.linalg.norm(r)
    q[:,0] = r / r_norm
    h = np.zeros((nmax_iter + 1, nmax_iter),dtype=b.dtype)
    return r, r_norm, q, h


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
        x_pcpg, rk , proj_r_hist, X_hist, info_dict = PCPG(A.dot,r,max_int=6)
        np.testing.assert_array_almost_equal(x_target,x_pcpg,decimal=10)

    def test_gmres_complex(self):
        K1 = sparse.csc_matrix(np.array([[2,-1,0],[-1,2,-1],[0,-1,1]]), dtype=np.float) 
        M1 = sparse.eye(K1.shape[0])
        f = np.array([0,0,1], dtype=np.complex)

        alpha = 0.01
        beta = 0.001
        Z_func = lambda w : K1 - w**2*M1 + 1J*w*(alpha*K1 + beta*M1)   

        Z = Z_func(0.5)
        lu = sparse.linalg.splu(Z)
        u_target = lu.solve(f)
        
        u_cg, info = sparse.linalg.cg(Z,f)
        u_gmres_scipy, info = sparse.linalg.gmres(Z,f)

        u_pcpg, rk , proj_r_hist, X_hist, info_dict = PCPG(Z.dot,f,max_int=100)
        u_gmres = GMRes(Z,f,nmax_iter=100,tol=1.0e-9)

        u_pgmres, rk , proj_r_hist, X_hist, info_dict = PGMRES(lambda x: Z.dot(x),f,max_int=100)
        u_pgmres_, rk , proj_r_hist, X_hist, info_dict = PGMRES(lambda x: Z.dot(x),f,
                                                                Precondicioner_action=lambda x: lu.solve(x), 
                                                                max_int=100)

        np.testing.assert_array_almost_equal(u_target,u_gmres,decimal=8)
        np.testing.assert_array_almost_equal(u_gmres_scipy,u_gmres,decimal=8)
        np.testing.assert_array_almost_equal(u_target,u_pgmres,decimal=8)
        np.testing.assert_array_almost_equal(u_target,u_pgmres_,decimal=8)


    def test_gmres(self):
        
        K = sparse.csc_matrix(np.array([[0,1],[-1,0]]), dtype=np.float) 
        f = np.array([1,1], dtype=np.float)

        u_target = sparse.linalg.spsolve(K,f)
        
        u_cg, info = sparse.linalg.cg(K,f)
        u_gmres_scipy, info = sparse.linalg.gmres(K,f)

        u_pcpg, rk , proj_r_hist, X_hist, info_dict = PCPG(K.dot,f,max_int=100)
        u_gmres = GMRes(K,f,nmax_iter=100,tol=1.0e-9)

        u_pgmres, rk , proj_r_hist, X_hist, info_dict = PGMRES(lambda x: K.dot(x),f,max_int=100)

        np.testing.assert_array_almost_equal(u_target,u_gmres,decimal=8)
        np.testing.assert_array_almost_equal(u_gmres_scipy,u_gmres,decimal=8)
        np.testing.assert_array_almost_equal(u_target,u_pgmres,decimal=8)



if __name__ == '__main__':
    main()