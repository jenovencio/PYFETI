
import sys 
import numpy as np
from unittest import TestCase, main
from collections import OrderedDict
from scipy import sparse
from scipy import optimize
#from scipy.fftpack import rfft, irfft, fft, ifft
from numpy.fft import rfft, irfft, fft, ifft
import time, logging
import matplotlib.pyplot as plt
import numdifftools as nd
from scipy.sparse.linalg import LinearOperator
from scipy import interpolate 

from pyfeti.src.nonlinalg import NonLinearOperator
from pyfeti.src.nonlinear import NonLinearLocalProblem, NonlinearSolverManager 
from pyfeti.src.optimize import feti as FETIsolver
from pyfeti.src.optimize import newton
from contpy import optimize as copt

class intercont():
    p_array = np.array([])
    yn_array = np.array([])
    dpn_init = 0.05

    def __init__(self):
        pass
        self.forward = True
        

    def __call__(self,pn,yn):
        ''' this function updates
        a parameter variable p, based on
        previous pair values (pn,yn)  
        '''
        self.append(pn,yn)
        if len(intercont.p_array)<2:
            dpn = self.point()
        elif len(intercont.p_array)<4:
            dpn = self.secant()
        else:
            dpn = self.interpolate()
        
        return dpn

    def append(self,pn,yn):
        try:
            intercont.yn_array = np.vstack((intercont.yn_array,yn))
        except:
            intercont.yn_array = yn
        try:
            intercont.p_array = np.concatenate((intercont.p_array,np.array([pn])))
        except:
            intercont.p_array = np.concatenate((intercont.p_array,pn))

    def point(self):
        
        self.y_update = 0.0*intercont.yn_array
        return intercont.dpn_init

    def interpolate(self):
        #cs = interpolate.CubicSpline(np.real(intercont.p_array),intercont.yn_array)
        #dy = cs.derivative().__call__(np.real(intercont.p_array)[-1])

        #sign_dy =  np.sign(intercont.yn_array[-1,:] - intercont.yn_array[-2,:])

        return self.secant()

    def secant(self):
        dp = intercont.p_array[-1] - intercont.p_array[-2]
        dy =  intercont.yn_array[-1,:] - intercont.yn_array[-2,:]
        #dy = dy/np.linalg.norm(dy)
        aug_v = np.concatenate((dy,np.array([dp])))
        aug_v = aug_v/np.linalg.norm(aug_v)
        base_vector = np.zeros(aug_v.shape,dtype =aug_v.dtype)
        base_vector[-1] = 1.0
        dp = np.real(intercont.dpn_init*np.dot(aug_v,base_vector))
        sign_dy =  np.real(np.sign(dy))[0]

        
        
        if sign_dy<1:
            stoppp = 1

        if np.abs(dp)<intercont.dpn_init*1.e-3:
            stoppp = 1
            if self.forward:
                self.forward = not self.forward

        if self.forward:
            sign_dp = 1
        else:
            sign_dp = -1

        y_update = (aug_v - base_vector)[0:-1]
        self.y_update = y_update
        return sign_dp*dp



class  Test_NonlinearSolver(TestCase):
    def setup_1D_linear_localproblem(self,nH = 1, beta = 0.0, alpha = 0.0, fscale=1.0):
        '''
        setup a simple one 1 problem with 2
       linear domains

                F->            <-F
        |>------0------0  0------0-------<|

        Parameters:
            nH : int
                number of Harmonics to be considered
            beta : float
                Stiffness coefficient for linear Damping, C = alpha*M + beta*K
            alpha : float
                Mass coefficient for linear Damping, C = alpha*M + beta*K
        returns :
            Z1, Z2, B1, B2, fn1_, fn2_

        '''

        K1 = np.array([[2.0,-1.0],
                       [-1.0,1.0]])

        K2 = np.array([[1.0,-1.0],
                       [-1.0,2.0]])


        M1 = np.array([[1.0,0.0],
                       [0.0,1.0]])

        M2 = M1
        

        
        C1 = alpha*M1 + beta*K1
        C2 = alpha*M2 + beta*K2


        #f = np.array([1.0,3.0])
        
        B1 = {(1,2): np.kron(np.eye(nH),np.array([[0.0,1.0]]))}
        B2 = {(2,1): np.kron(np.eye(nH),np.array([[-1.0,0.0]]))}

        f1_ = fscale*np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), np.array([1.0,0.0]))
        f2_ = fscale*np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])),np.array([0.0,-1.0]))

        cfn1_ = lambda u_,w=np.zeros(nH) : -f1_
        cfn2_ = lambda u_,w=np.zeros(nH) : -f2_

        JZ1 =  lambda u_,w=np.zeros(nH) : np.kron(-np.diag(w**2),M1) + 1J*np.kron(np.diag(w**2),C1) + np.kron(np.eye(w.shape[0]),K1)
        JZ2 =  lambda u_,w=np.zeros(nH) : np.kron(-np.diag(w**2),M2) + 1J*np.kron(np.diag(w**2),C2) + np.kron(np.eye(w.shape[0]),K2)

        callback_func = lambda u_,w=np.zeros(nH) : JZ1(u_,w).dot(u_)
        Z1 = NonLinearOperator(callback_func , shape=JZ1(0).shape, jac=JZ1)
        fn1_ = NonLinearOperator(cfn1_ , shape=JZ1(0).shape, jac=np.zeros(JZ1(0).shape))
        
        callback_func2 = lambda u_,w=np.zeros(nH) : JZ2(u_,w).dot(u_)
        Z2 = NonLinearOperator(callback_func2 , shape=JZ2(0).shape, jac=JZ2)
        fn2_ = NonLinearOperator(cfn2_ , shape=JZ2(0).shape, jac=np.zeros(JZ2(0).shape))

       
        return Z1, Z2,B1, B2, fn1_, fn2_

    def setup_1D_nonlinear_localproblem(self,nH = 1, beta = 0.0, alpha = 0.0, c = 0.0, fscale=1.0):
        '''
        setup a simple one 1 problem with 2
        linear domains

                   F->            <-F
        |>-*---*---0------0  0------0-------<|
           |/\/|                        
       nonlinear spring            

        Parameters:
            nH : int
                number of Harmonics to be considered
            beta : float
                Stiffness coefficient for linear Damping, C = alpha*M + beta*K
            alpha : float
                Mass coefficient for linear Damping, C = alpha*M + beta*K
            c  : float
                nonlinear spring coefficinet

        returns :
            Z1, Z2, B1, B2, fn1_, fn2_

        '''

        ndof = 2
        Z1, Z2,B1, B2, fn1_, fn2_ = self.setup_1D_linear_localproblem(nH,beta,alpha,fscale=fscale)


        f1_ = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), np.array([1.0,0.0]))
        f2_ = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])),np.array([0.0,-1.0]))

        # FFT time domain to freq, iFFT freq to time
        
        mode  = 'ortho'
        FFT = lambda u : rfft(u,norm=mode).T[0:nH+1].reshape((nH+1)*ndof,1).flatten()[ndof:] # removing the static part
        iFFT = lambda u_ : 2.0*np.real(ifft(np.concatenate((np.zeros(ndof),u_)).reshape(nH+1,ndof).T, n=100,norm=mode))

        # nonlinear force in Time
        fnl = lambda u, n=3 : c*np.array([u[0]**n, u[1]*0.0])
        fnl_ = lambda u_, n=3 : FFT(fnl(iFFT(u_),n))

        u_ = np.array([1.0]*ndof*nH, dtype=np.complex)
        u_[1:] = 0.0
        #u_[3:] = 0.0
        #u_[0:2] = 0.0
        
        np.testing.assert_almost_equal(u_, FFT(iFFT(u_)), decimal=8)

        cfn1_ = lambda u_, w=np.zeros(nH) : -f1_ + fnl_(u_)
        
        fn1_ = NonLinearOperator(cfn1_ , shape=Z1.shape, jac=np.zeros(Z1.shape))
        

        return Z1, Z2,B1, B2, fn1_, fn2_
        
    def test_1D_linear_localproblem(self):

        nH = 2 # number of Harmonics
        Z1, Z2,B1, B2, fn1_, fn2_ = self.setup_1D_linear_localproblem(nH)

        JZ1 = Z1.jac

        length = fn1_.shape[0]
        nonlin_obj1 = NonLinearLocalProblem(Z1,B1,fn1_,length)
        nonlin_obj2 = NonLinearLocalProblem(Z2,B2,fn2_,length)

        # Defining a Harmonic Lambda = 0
        lambda_dict = {}
        lambda_dict[(1,2)] = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), np.array([0.0]))
        lambda_dict[(2,1)] = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), np.array([0.0]))

        freq_list = np.arange(0.0,1.0,0.01)
        u_target_1 = np.zeros(freq_list.shape[0],dtype=np.complex)
        u_calc_1 = np.zeros(freq_list.shape[0],dtype=np.complex)
        
        for i,freq in enumerate(freq_list):
            w = 2.0*np.pi*freq*np.arange(1,nH+1)

            # using linear solver
            ui_ = np.linalg.solve(JZ1(0,w),-fn1_.eval(0,w))
            u_target_1[i] = ui_[0]

            # using nonlinear solver 
            ui_calc = nonlin_obj1.solve(lambda_dict,w)
            u_calc_1[i] = ui_calc[0]

        np.testing.assert_almost_equal(u_target_1, u_calc_1, decimal=8)
        
        # plotting frequency response
        if False:            
            ax = plt.axes()
            ax.plot(np.abs(u_target_1),'r--',label='target')
            ax.plot(np.abs(u_calc_1),'b*',label='calc')
            plt.legend()
            plt.show()

    def l_array2dict(self,l):
            nH = self.nH
            lambda_dict = {}
            #lambda_dict[(1,2)] = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), l)
            #lambda_dict[(2,1)] = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), l)
            lambda_dict[(1,2)] = l
            lambda_dict[(2,1)] = l
            return  lambda_dict

    def test_1D_linear_localproblem_2(self):
        nH = 1
        Z1, Z2,B1, B2, fn1_, fn2_ = self.setup_1D_linear_localproblem(nH,beta=0.1)
        length = fn1_.shape[0]
        nonlin_obj1 = NonLinearLocalProblem(Z1,B1,fn1_,length)
        nonlin_obj2 = NonLinearLocalProblem(Z2,B2,fn2_,length)


        int_force_list = np.arange(0.0,1.0,1.0)
        freq_list = np.arange(0.0,0.5,0.02)
        u_calc_list = []
        u_calc_list_2 = []
        r_list =[]
        for freq in freq_list:
            w = 2.0*np.pi*freq*np.arange(1,nH+1)
            u_calc_1 = []
            u_calc_2 = []
            r_calc = []
            
            for int_force in int_force_list:

                lambda_dict = {}
                lambda_dict[(1,2)] = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), np.array([int_force]))
                lambda_dict[(2,1)] = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), np.array([int_force]))
            
                # using nonlinear solver 
                ui_calc = nonlin_obj1.solve(lambda_dict,w)
                uj_calc = nonlin_obj2.solve(lambda_dict,w)
                if (ui_calc is not None) and (uj_calc is not None):
                    u_calc_1.append(ui_calc[1])
                    u_calc_2.append(uj_calc[0])
                    r_calc.append(np.linalg.norm(B1[1,2].dot(ui_calc) + B2[2,1].dot(uj_calc)))
                #else:
                #    u_calc_1, u_calc_2, r_calc = None, None, None

                u_calc_list.append(u_calc_1)
                u_calc_list_2.append(u_calc_2)
                r_list.append(r_calc)

        # plotting forced responde varing interface force (lambda)                        
        if False:            
            ax = plt.axes()
            for int_force, u_calc in zip(int_force_list,u_calc_list):
                ax.plot(np.abs(u_calc),'ro', label=('D1, $\lambda$ = %2.2e' %int_force))

            for int_force, u_calc in zip(int_force_list,u_calc_list_2):
                ax.plot(np.abs(u_calc),'b*', label=('D2, $\lambda$ = %2.2e' %int_force))
            
            plt.legend()
            plt.show()


        # plotting interface residual varing interface force (lambda) and frequency                       
        if False:  
            ax = plt.axes()
            lambda_list = []
            min_r_list = []
            for freq, r in zip(freq_list,r_list):
                min_r_id = np.argmin(r)
                fb = int_force_list[min_r_id]
                lambda_list.append(fb)
                min_r_list.append(r[min_r_id])
            ax.plot(freq_list,lambda_list,'--')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('$\lambda$ [N]')

            
            fig2, ax1 = plt.subplots(1,1)
            ax1.plot(freq_list,min_r_list,'--')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('$\Delta u$ [mm]')

            
            plt.show()

    def test_1D_linear_dual_interface_problem(self):
        ''' Test linear problem with no Damping
        '''
        nH,c,beta,alpha = 1, 0.0, 0.0, 0.0
        Rb,nc,nH, nonlin_obj_list,JRb = self.setup_nonlinear_problem(nH,c,beta,alpha) 
        self.run_dual_interface_problem(Rb,nc,nH, nonlin_obj_list,jac=JRb)

    def test_1D_linear_dual_interface_nonlinear_problem(self):
        ''' Test nonlinear problem with Damping
        '''
        nH,c,beta,alpha = 1, 3.0, 1.8, 0.0
        Rb,nc,nH, nonlin_obj_list,JRb = self.setup_nonlinear_problem(nH,c,beta,alpha) 
        self.run_dual_interface_problem(Rb,nc,nH, nonlin_obj_list,jac=JRb)
  
    def setup_nonlinear_problem(self,nH=1,c=0.0,beta=0.18,alpha=0.0,fscale=1.0):
        
       
        self.nH = nH
        Z1, Z2,B1, B2, fn1_, fn2_ = self.setup_1D_nonlinear_localproblem(nH,c=c, beta = beta, alpha=alpha, fscale=fscale)
        length = fn1_.shape[0]
        nonlin_obj1 = NonLinearLocalProblem(Z1,B1,fn1_,length)
        nonlin_obj2 = NonLinearLocalProblem(Z2,B2,fn2_,length)

        # defining the Residual at the interface
        Rb_ = lambda w0 : lambda l : B1[1,2].dot(nonlin_obj1.solve(self.l_array2dict(l),w0)) + \
                                    B2[2,1].dot(nonlin_obj2.solve(self.l_array2dict(l),w0))        

        Rb = lambda w0 : lambda l : nonlin_obj1.solve_interface_displacement(self.l_array2dict(l),w0)[1,2] + \
                                    nonlin_obj2.solve_interface_displacement(self.l_array2dict(l),w0)[2,1]

        JRb = lambda w0 : lambda l : nonlin_obj1.derivative_u_over_lambda(w0)[1,2] + \
                                    nonlin_obj2.derivative_u_over_lambda(w0)[2,1]

        w = 0.0
        Rl= Rb(w)
        Rl_= Rb_(w)
        nc = B1[1,2].shape[0]
        Rb = Rb_
        l0 = np.ones(nc, dtype=np.complex)
        l0 = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), l0)
        
        nonlin_obj_list = [nonlin_obj1, nonlin_obj2]
    
        return Rb,nc,nH, nonlin_obj_list,JRb

    def test_compare_FETI_vs_explicit_inverse(self):
        ''' The Dual Nonlinear problem need to solve the linear system of the newton iteration

        F * delta_l = error

        F can be explicit assembled, but it needs a matrix mpi communication
        otherwise, a FETI solver can be use to solve Newton iteration

        '''
        nH,c,beta,alpha = 1, 3.0, 1.8, 0.0
        Rb,nc,nH, nonlin_obj_list,JRb = self.setup_nonlinear_problem(nH,c,beta,alpha) 
        

        f = 0.0
        w0 = w = 2.0*np.pi*f*np.arange(1,nH+1)
        tol = 1.0e-8
        Rl= Rb(w)
        JRl = JRb(w)
        l0 = np.ones(nc, dtype=np.complex)


        lambda_dict = self.l_array2dict(l0)
        nonlinear_problem_dict = {}
        nonlinear_problem_dict[1] = nonlin_obj_list[0]
        nonlinear_problem_dict[2] = nonlin_obj_list[1]
        local_problem_dict = {}
        for key, nonlinear_obj in nonlinear_problem_dict.items():
            local_problem_dict[key] = nonlinear_obj.build_linear_problem(lambda_dict,w0,u=None)

        r0 = Rl(l0)
        x_target = np.linalg.solve(JRl(l0),r0)
        x = FETIsolver(local_problem_dict)

        e = np.abs(x - x_target)
        np.testing.assert_array_almost_equal(x,x_target,decimal=10)
        
        x=1
            
    def run_dual_interface_problem(self,Rb,nc,nH, nonlin_obj_list = [],jac=None):

        try:
            nonlin_obj1  = nonlin_obj_list[0]
            nonlin_obj2 = nonlin_obj_list[1]
        except:
            pass
        
        lambda_list = []
        min_r_list = []
        freq_list = []
        u1_list = []
        u2_list = []
        l0 = np.zeros(nc, dtype=np.complex)
        freq_init = 0.0
        delta_freq = 0.01
        n_int = 150
        default_scalling = 1
        scalling = 1
        factor = 0.9
        freq = freq_init
        count = 0
        forward = True
        jump = True
        for n in range(n_int):
            w = 2.0*np.pi*freq*np.arange(1,nH+1)
            tol = 1.0e-8
        
            Rl= Rb(w)
            
            sol = None
            try:

                #sol = optimize.root(Rl, l0, method='lm', jac=JRl_num, options={'fatol': tol, 'maxiter' : 20})
                #sol = optimize.root(Rl, l0, method='krylov', options={'fatol': tol, 'maxiter' : 20})
                JRl = jac(w)
                sol = newton(Rl,JRl,l0)
                #sol = optimize.root(Rl, l0, method='lm', options={'fatol': tol, 'maxiter' : 20})
                print('Number of iterations %i' %sol.nit)
                if sol.success:
                    # restart success counter
                    count = 0
                    scalling = default_scalling
                    l0 =  sol.x
                    r_vec = sol.fun
                    r = np.linalg.norm(r_vec )
                    np.testing.assert_almost_equal(r, 0.0, decimal=8)

                    min_r_list.append(r)
                    lambda_list.append(l0)

                    freq_list.append(freq)
                    u1 = nonlin_obj1.u_init
                    u2 = nonlin_obj2.u_init

                    u1_list.append(u1)
                    u2_list.append(u2)

                    
                    JLO = lambda l : LinearOperator(shape=(1,1), dtype=np.complex, matvec = lambda v : JRl(l).dot(v))
                    #JRl_num = nd.Jacobian(Rl,n=1)

                    JRl_eval = JRl(l0)
                    #JRl_num_eval = JRl_num(l0)
                    #np.testing.assert_array_almost_equal(JRl_eval,JRl_num_eval,decimal=10)
                else:
                    raise Exception
            except:
                count +=1
                 
                print('Interface Problem did not converge! Try number %i' %count)
                if count>3:
                    # jump
                    if jump:
                        freq += 15*delta_freq*default_scalling
                        freq_jump = freq
                        jump=False
                    else:
                        freq = freq_jump
                        jump = True

                    print('Frequency jump = %2.2e' %freq_jump)
                    # go backwards
                    if forward:
                        factor=0.9
                        scalling = -default_scalling
                        forward=False
                    #go forward again
                    else:
                        factor = 1.0
                        scalling = default_scalling
                        forward=True
                    count = 0
                    
                else:
                    freq = freq_list[-1]
                scalling = scalling*factor


            freq += delta_freq*scalling
            
        plot_results = False
        if plot_results:
            fig1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
            ax1.plot(freq_list,np.abs(lambda_list),'*--')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('$\lambda$ [N]')

            #fig2, ax2 = plt.subplots(1,1)
            ax2.plot(freq_list,min_r_list,'*--')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_ylabel('$\Delta u$ [mm]')
            

            #fig3, ax3 = plt.subplots(1,1)
            ax3.plot(freq_list,np.abs(u1_list),'*--')
            ax3.set_xlabel('Frequency [Hz]')
            ax3.set_ylabel(' u1 [mm]')

            #fig4, ax4 = plt.subplots(1,1)
            ax4.plot(freq_list,np.abs(u2_list),'*--')
            ax4.set_xlabel('Frequency [Hz]')
            ax4.set_ylabel(' u2 [mm]')
            
            plt.show()

    def Test_NonlinearSolverManager(self):
        nH,c,beta,alpha = 1, 0.0, 1.0, 0.0
        Z1, Z2,B1, B2, fn1_, fn2_ = self.setup_1D_nonlinear_localproblem(nH,c=c, beta = beta, alpha=alpha)
        Z_dict = {1:Z1,2:Z2}
        B_dict = {1:B1,2:B2}
        f_dict = {1:fn1_,2:fn2_}
        manager = NonlinearSolverManager(Z_dict,B_dict,f_dict)
        manager.build_local_to_global_mapping()
        manager.lambda_init = np.zeros(manager.lambda_size, dtype=manager.dtype)
        

        freq_list = np.arange(0,0.5,0.01)
        lambda_list = []
        for freq in freq_list:
            w0 = 2.0*np.pi*freq
            try:
                sol = manager.solve_dual_interface_problem(w0=np.array([w0]))
                lambda_sol = sol.x
                manager.lambda_init = lambda_sol
            except:
                sol = manager.solve_dual_interface_problem(w0=np.array([w0]))
                lambda_sol = 0.0
            lambda_list.append(lambda_sol)
            

        plt.plot(freq_list,np.abs(lambda_list),'o')
        plt.show()
        
        x=1

    def test_intercont(self):
        ''' test interpolation continuation
        '''

        s = 2.0*np.pi
        f = lambda p : np.array([np.cos(s*p), np.sin(s*p)])

        p = 0.0
        dp = 0.05
        #y_target = np.array([[]])
        p_array = np.array([])
        for i in range(50):
            y = f(p)
            try:
                y_target = np.vstack((y_target,y))
            except:
                y_target = y
            p_array = np.concatenate((p_array,np.array([p])))
            p+=dp

        cont = intercont()
        p = 0.0
        for i in range(50):
            dp = cont(p,f(p))
            p+=dp
            


        plt.plot(y_target.T[0,:],y_target.T[1,:],'o')
        plt.plot(cont.yn_array.T[0,:],cont.yn_array.T[1,:],'*')
        plt.show()

        x=1

    def test_intercont_1d_Duffing(self):

        a = 1.0 #5.e0
        b = 1
        nH = 1
        ndof = 1
        d = 0.1
        mode  = 'ortho'
        FFT = lambda u : rfft(u,norm=mode).T[0:nH+1].reshape((nH+1)*ndof,1).flatten()[ndof:] # removing the static part
        iFFT = lambda u_ : 2.0*np.real(ifft(np.concatenate((np.zeros(ndof),u_)).reshape(nH+1,ndof).T, n=100,norm=mode))

        # nonlinear force in Time
        fnl = lambda x, n=3 : a*x**n
        dfnl = lambda x, n=3 : n*a*x**(n-1)
        fnl_ = lambda x, n = 3 : FFT(fnl(iFFT(np.array([x])),n))[0]
        dfnl_ = lambda w : lambda x, n = 3 : np.array([[FFT(dfnl(iFFT(x),n))[0]]])

        f = lambda x,w : -w**2*x[0] + x[0] + d*1J*w*x[0] + fnl_(x[0]) - b
        dfx = lambda w : lambda x : -w**2 + 1.0 + d*1J*w +  + dfnl_(w)(x)
        x_implicit = lambda w, x_init : optimize.root(lambda x : f(x,w),x0=x_init,method='krylov').x

        dfx_num = lambda w : lambda x : nd.Jacobian(lambda x : f(x,w),n=1)(x)
        dfx_num_ = lambda w : lambda x : copt.complex_jacobian(lambda x : f(x,w),n=1)(x)

        p = 0.01
        x = np.array([0.01],dtype=np.complex)

        p_range=(0.0,1.5)
        start_time = time.time()
        x_sol, p_sol, info_dict = copt.continuation(f,x0=x,p_range=p_range,p0=0.0, step=0.3,jacx=dfx_num_) # 
        elapsed_time = time.time() - start_time
        print('{"Continuation elapsed time" : %f} #Elapsed time (s)' %elapsed_time)
        plt.plot(p_sol,np.abs(x_sol)[0,:],'o')
        plt.show()

    def test_linear_freq_response_cont(self):

        nH,c,beta,alpha = 1, 1.0e-0, 1.0, 0.0
        fscale = 1.0
        Rb,nc,nH, nonlin_obj_list,JRb = self.setup_nonlinear_problem(nH,c,beta,alpha,fscale=fscale) 
        
        

        
        
        tol = 1.0e-8
        scale = 0.8
        l0 = np.ones(nc, dtype=np.complex)
        
        map_dict = {}
        map_dict[(1,2)] = list(range(len(l0)))
        map_dict[(2,1)] = list(range(len(l0)))
        nl1 = nonlin_obj_list[0]
        nl1.map = map_dict


        p_range=(0.1,0.5)
        R = lambda l, w : Rb(np.array([w]))(l)
        JR = lambda w : lambda l : JRb(np.array([w]))(l)
        start_time = time.time()
        x_sol, p_sol, info_dict = copt.continuation(R,x0=l0,p_range=p_range,step=0.05,jacx=JR,correction_method='matcont')
        elapsed_time = time.time() - start_time
        print('{"Continuation elapsed time" : %f} #Elapsed time (s)' %elapsed_time)
        
        
                
        u1_list = []
        for w, l in zip(p_sol,x_sol.T):
            u1 = nl1.solve_displacement(l,np.array([w]))
            if u1 is None:
                u1 = 0.0*u1_list[-1]
            
            u1_list.append(u1)


        plt.plot(p_sol,np.abs(x_sol)[0,:],'o')
        plt.title('$\lambda$')
        

        plt.figure()
        plt.plot(p_sol, np.abs(np.array(u1_list)),'*')
        plt.title('displacement 1')
        plt.show()

        x=1
        
        
    def _test_NonLinearLocalProblem(self):

        nH,c,beta,alpha = 1, 00.e-0, 1.0, 0.0
        fscale = 1.0
        Rb,nc,nH, nonlin_obj_list,JRb = self.setup_nonlinear_problem(nH,c,beta,alpha,fscale=fscale) 

        l0 = np.ones(nc, dtype=np.complex)
        nl1 = nonlin_obj_list[0]
        map_dict = {}
        map_dict[(1,2)] = list(range(len(l0)))
        map_dict[(2,1)] = list(range(len(l0)))
        nl1.map = map_dict

        R = nl1.local_equilibrium_equation()
       






if __name__=='__main__':

    #main()
    testobj = Test_NonlinearSolver()
    #testobj.test_1D_linear_localproblem()
    #testobj.test_1D_linear_dual_interface_problem()
    #testobj.setup_1D_nonlinear_localproblem()
    #testobj.test_1D_linear_dual_interface_nonlinear_problem()
    #testobj.test_compare_FETI_vs_explicit_inverse()
    #testobj.test_1D_linear_localproblem_2()
    #testobj.Test_NonlinearSolverManager()
    #testobj.test_intercont()
    #testobj.test_linear_freq_response_cont()
    testobj.test_intercont_1d_Duffing()
    #testobj.test_NonLinearLocalProblem()