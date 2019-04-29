
import sys 
import numpy as np
from unittest import TestCase, main
from collections import OrderedDict
from scipy import sparse
from scipy import optimize
#from scipy.fftpack import rfft, irfft, fft, ifft
from numpy.fft import rfft, irfft, fft, ifft
import time
import matplotlib.pyplot as plt

from pyfeti.src.nonlinalg import NonLinearOperator
from pyfeti.src.nonlinear import NonLinearLocalProblem


class  Test_NonlinearSolver(TestCase):
    def setup_1D_linear_localproblem(self,nH = 1, beta = 0.0, alpha = 0.0):
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


 
        f1_ = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), np.array([1.0,0.0]))
        f2_ = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])),np.array([0.0,-1.0]))

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

    def setup_1D_nonlinear_localproblem(self,nH = 1, beta = 0.0, alpha = 0.0, c = 0.0):
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
        Z1, Z2,B1, B2, fn1_, fn2_ = self.setup_1D_linear_localproblem(nH)


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
            lambda_dict[(1,2)] = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), l)
            lambda_dict[(2,1)] = np.kron(np.concatenate(([1.0,],(nH-1)*[0.0])), l)
            return  lambda_dict

    def test_1D_linear_localproblem_2(self):
        nH = 1
        Z1, Z2,B1, B2, fn1_, fn2_ = self.setup_1D_linear_localproblem(nH)
        length = fn1_.shape[0]
        nonlin_obj1 = NonLinearLocalProblem(Z1,B1,fn1_,length)
        nonlin_obj2 = NonLinearLocalProblem(Z2,B2,fn2_,length)


        int_force_list = np.arange(0.0,1.0,0.05)
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
                u_calc_1.append(ui_calc[1])
                u_calc_2.append(uj_calc[0])
                r_calc.append(np.linalg.norm(B1[1,2].dot(ui_calc) + B2[2,1].dot(uj_calc)))
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

        nH = 1
        self.nH = nH
        Z1, Z2,B1, B2, fn1_, fn2_ = self.setup_1D_linear_localproblem(nH)
        length = fn1_.shape[0]
        nonlin_obj1 = NonLinearLocalProblem(Z1,B1,fn1_,length)
        nonlin_obj2 = NonLinearLocalProblem(Z2,B2,fn2_,length)

        # defining the Residual at the interface
        Rb = lambda w0 : lambda l : B1[1,2].dot(nonlin_obj1.solve(self.l_array2dict(l),w0)) + \
                                    B2[2,1].dot(nonlin_obj2.solve(self.l_array2dict(l),w0))        
        nc = B1[1,2].shape[0]

        self.run_dual_interface_problem(Rb,nc,nH)

    def test_1D_linear_dual_interface_nonlinear_problem(self):
        
        nH = 1
        self.nH = nH
        Z1, Z2,B1, B2, fn1_, fn2_ = self.setup_1D_nonlinear_localproblem(nH,c=1.0e-2)
        length = fn1_.shape[0]
        nonlin_obj1 = NonLinearLocalProblem(Z1,B1,fn1_,length)
        nonlin_obj2 = NonLinearLocalProblem(Z2,B2,fn2_,length)

        # defining the Residual at the interface
        Rb = lambda w0 : lambda l : B1[1,2].dot(nonlin_obj1.solve(self.l_array2dict(l),w0)) + \
                                    B2[2,1].dot(nonlin_obj2.solve(self.l_array2dict(l),w0))        
        nc = B1[1,2].shape[0]

        nonlin_obj_list = [nonlin_obj1, nonlin_obj2]
        self.run_dual_interface_problem(Rb,nc,nH, nonlin_obj_list)

    def run_dual_interface_problem(self,Rb,nc,nH,nonlin_obj_list = []):
        
        try:
            nonlin_obj1  = nonlin_obj_list[0]
            nonlin_obj2 = nonlin_obj_list[1]
        except:
            pass
        
        lambda_list = []
        min_r_list = []
        freq_list = np.arange(0.0,0.5,0.01)
        freq = freq_list[0]
        u1_list = []
        u2_list = []
        l0 = np.zeros(nc, dtype=np.complex)
        for freq in freq_list:
            w = 2.0*np.pi*freq*np.arange(1,nH+1)
            tol = 1.0e-8
            
            Rl = Rb(w)
            
            try:
                sol = optimize.root(Rl, l0, method='krylov', options={'fatol': tol})
            except:
                sol.fun = 1.0e-8
                sol.x = 0.0

            r = np.linalg.norm(sol.fun)
            np.testing.assert_almost_equal(r, 0.0, decimal=8)
            min_r_list.append(r)
            lsol = sol.x
            l0 = lsol
            lambda_list.append(lsol)

            u1 = nonlin_obj1.solve(self.l_array2dict(lsol),w)
            u2 = nonlin_obj2.solve(self.l_array2dict(lsol),w)

            u1_list.append(u1)
            u2_list.append(u2)

            
        plot_results = True
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



        x =1



if __name__=='__main__':

    #main()
    testobj = Test_NonlinearSolver()
    #testobj.test_1D_linear_dual_interface_problem()
    #testobj.setup_1D_nonlinear_localproblem()
    testobj.test_1D_linear_dual_interface_nonlinear_problem()