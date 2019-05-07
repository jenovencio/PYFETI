import numpy as np
from unittest import TestCase, main


class NonLinearOperator():
    def __init__(self,callback, jac=None, shape =None, dtype=None):
        self.callback = callback
        self._jac = jac
        self._shape = shape
        self.dtype = dtype
        self._kernel = np.array([])
       
    def __call__(self,*args,**kargs):
        return self.eval(*args,**kargs)

    @property
    def shape(self,):
        return self._shape

    @shape.setter
    def shape(self,shape):
        self._shape = shape

    @property
    def jac(self):
        return self._jac

    @jac.setter
    def jac(self,J):
        self._jac = J

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self,R):
        self._kernel = R

    def eval(self,*args,**kargs):
        return self.callback(*args,**kargs)

    def linearize(self,*args,**kargs):
        return self._jac(*args,**kargs)



    
class  Test_Nonlinalg(TestCase):
    def test_nonlinear_operator(self):
        
        K = np.array([[2.0,-1.0],
                       [-1.0,1.0]])

        M = np.array([[1.0,0.0],
                       [0.0,1.0]])

        
        JZu =  lambda u_,w=0.0 : w*w*M + K

        Z = NonLinearOperator(lambda u_,w=0 : JZu(u_,w).dot(u_), shape=K.shape, jac=JZu)
        
        w = 4.0
        u_ =  np.array([1.0,1.0])
        np.testing.assert_equal(Z.linearize(u_,w),JZu(u_,w))
        np.testing.assert_equal(Z.eval(u_,w), JZu(u_,w).dot(u_))
        np.testing.assert_equal(Z.shape, K.shape)

if __name__=='__main__':
    main()