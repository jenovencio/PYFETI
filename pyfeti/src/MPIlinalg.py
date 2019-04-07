import numpy as np
from scipy import sparse
from scipy import linalg
import os
from unittest import TestCase, main
import logging
import time
from mpi4py import MPI

def get_chunks(number_of_chuncks,size):
    ''' create chuncks based on number of mpi process
    '''

    n = size
    remaining = n%number_of_chuncks
    default_size =  int((n-remaining)/number_of_chuncks)
    chunck_sizes = np.linspace(default_size,n,number_of_chuncks,dtype=np.int)
    chunck_sizes[-1] += remaining
    return chunck_sizes


class ParallelMatrix():
    '''
    Paramenters
        A : csr sparse matrix
            matrix to store in parallel
        n : int
            number of chunks to split matrix
        isserialize : Boolean 
            True is the matrix is serialized 
    '''
    
    def __init__(self,A, n,tmp_dir = 'tmp',prefix = 'A_'):
        self.A = A
        self.n = n
        self.tmp_dir = tmp_dir
        self.prefix = prefix
        self.ext = 'npz'
        self.isserialize = False
        self.get_chunks()

    def get_chunks(self):
        ''' create chuncks based on number of mpi process
        '''
        self.chunck_sizes = get_chunks(self.n,self.A.shape[0])
        return self.chunck_sizes

    def columns_serialization(self):
        try:
            os.mkdir(self.tmp_dir)
        except:
            pass

        init_id = 0
        for i,col_id in enumerate(self.chunck_sizes):
            file = os.path.join(self.tmp_dir,self.prefix + str(i) + '.' + self.ext)        
            logging.info('Saving maitrix to %s' %file)
            sparse.save_npz(file, self.A[:,init_id:col_id], compressed=True)
            init_id = col_id

        self.isserialize = True

    def load_columns_matrix(self,rank):
         file = os.path.join(self.tmp_dir,self.prefix + str(rank) + '.' + self.ext)        
         return sparse.load_npz(file)

class ParallelVector():
    '''
    Paramenters
        v : csr sparse matrix
            matrix to store in parallel
        n : int
            number of chunks to split matrix
        isserialize : Boolean 
            True is the vector is serialized

       False
    '''
    def __init__(self,v,n,tmp_dir = 'tmp',prefix = 'v_'):
        self.v = v
        self.n = n
        self.isserialize = False
        self.tmp_dir =  tmp_dir
        self.prefix = prefix
        self.ext = 'npy'
        self.get_chunks()
         

    def get_chunks(self):
        ''' create chuncks based on number of mpi process
        '''
        self.chunck_sizes = get_chunks(self.n,self.v.shape[0])
        return self.chunck_sizes

    def serialize(self):
        try:
            os.mkdir(self.tmp_dir)
        except:
            pass

        init_id = 0
        for i,col_id in enumerate(self.chunck_sizes):
            
            filearr = os.path.join(self.tmp_dir,self.prefix + str(i)  + '.' + self.ext)
            np.save(filearr, self.v[init_id:col_id])

            init_id = col_id

        self.isserialize = True

    def load_vector_chunck(self,rank):
         filearr = os.path.join(self.tmp_dir,self.prefix + str(rank)  + '.' + self.ext)   
         return np.load(filearr)

def matvec(A,v,n=2):
    ''' Parallel matrix multiplication
    u = A.dot(v)

    A :  csr sparse matrix
        sparse matrix 
    v : np.array

    
    '''

    parallel_matrix_obj = ParallelMatrix(A,n)
    parallel_matrix_obj.columns_serialization()
    parallel_vec_obj = ParallelVector(v,n)
    parallel_vec_obj.serialize()

    rank_id = 0

    u = np.zeros(v.shape)
    for rank_id in range(n):
        Acol = parallel_matrix_obj.load_columns_matrix(rank_id)
        vc =  parallel_vec_obj.load_vector_chunck(rank_id)
        u += Acol.dot(vc)


    return u



class  Test_FETIsolver(TestCase):
    def test_parallel_matvec(self):

        n = 10
        x = 3*np.arange(n)


        top = [4,-1]
        top.extend([0.0]*(n-2))
        A = linalg.toeplitz(top,top)
        A[n-1,0] = -1
        A[0,n-1] = -1
        A = sparse.csc_matrix(A)


        u_target = A.dot(x)

        u = matvec(A,x,n=2)

        np.testing.assert_almost_equal(u_target,u,decimal=10)




if __name__=='__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size==1:
        main()

    else:
   # execute only if run as a script
        args = []
        for s in sys.argv:
            args.append(s)    
        
        for arg in args:
            try:
               var, value = arg.split('=')
            except:
                print('Commnad line argument nor understoop, arg = %s cannot be splited in variable name + value' %arg)
