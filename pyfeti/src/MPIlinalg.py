import numpy as np
from scipy import sparse
from scipy import linalg
import os, sys, shutil
from unittest import TestCase, main
import logging
import time
from mpi4py import MPI
from utils import MPILauncher, getattr_mpi_attributes, pyfeti_dir

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
    
    def __init__(self,A, n,tmp_dir = 'tmp',prefix = 'A_',serializeit=True):
        self.A = A
        self.n = n
        self.tmp_dir = tmp_dir
        self.prefix = prefix
        self.ext = 'npz'
        self.isserialize = False
        self.get_chunks()

        if serializeit:
            self.columns_serialization()

    def get_chunks(self):
        ''' create chuncks based on number of mpi process
        '''
        self.chunck_sizes = get_chunks(self.n,self.A.shape[0])
        return self.chunck_sizes

    def columns_serialization(self):

        start_time = time.time()
        if not self.isserialize:
            try:
                os.mkdir(self.tmp_dir)
            except:
                pass

            init_id = 0
            for i,col_id in enumerate(self.chunck_sizes):
                file = os.path.join(self.tmp_dir,self.prefix + str(i) + '.' + self.ext)        
                sparse.save_npz(file, self.A[:,init_id:col_id], compressed=True)
                init_id = col_id

            self.isserialize = True
        
        elapsed_time = time.time() - start_time
        logging.info('Matrix serialization, Elapsed time : %4.5f ' %elapsed_time)
        return None

    def load_columns_matrix(self,rank):
         file = os.path.join(self.tmp_dir,self.prefix + str(rank) + '.' + self.ext)        
         return sparse.load_npz(file)

    def dot(self,v):

        parallel_vec_obj = ParallelVector(v,self.n)

        u = parallel_launcher(self.n,
                         module = 'MPIlinalg',
                         method = 'parallel_matvec',
                         tmp_dir = self.tmp_dir,
                         prefix_matrix = self.prefix,
                         ext_matrix =  self.ext,
                         prefix_array = parallel_vec_obj.prefix,
                         ext_array = parallel_vec_obj.ext)

        return u


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
    def __init__(self,v,n,tmp_dir = 'tmp',prefix = 'v_',serializeit=True):
        self.v = v
        self.n = n
        self.isserialize = False
        self.tmp_dir =  tmp_dir
        self.prefix = prefix
        self.ext = 'npy'
        self.get_chunks()
        if serializeit:
            self.serialize()
         
    def get_chunks(self):
        ''' create chuncks based on number of mpi process
        '''
        self.chunck_sizes = get_chunks(self.n,self.v.shape[0])
        return self.chunck_sizes

    def serialize(self):

        start_time = time.time()
        if not self.isserialize:
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

        elapsed_time = time.time() - start_time
        logging.info('vector serialization, Elapsed time : %4.5f ' %elapsed_time)
        return None

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

    if n==1:
        u = A.dot(v)
    else:

        parallel_matrix_obj = ParallelMatrix(A,n)
        parallel_vec_obj = ParallelVector(v,n)

        u = parallel_launcher(n,
                         module = 'MPIlinalg',
                         method = 'parallel_matvec',
                         tmp_dir = parallel_matrix_obj.tmp_dir,
                         prefix_matrix = parallel_matrix_obj.prefix,
                         ext_matrix =  parallel_matrix_obj.ext,
                         prefix_array = parallel_vec_obj.prefix,
                         ext_array = parallel_vec_obj.ext)

    return u

def parallel_launcher(n=2,**kwargs):
    # execute only if run as a script
    python_file = pyfeti_dir(r'src\MPIlinalg.py')
    mpi_size = n
    
    mip_obj = MPILauncher(python_file,mpi_size,**kwargs)
    mip_obj.run()

    filearr = os.path.join(kwargs['tmp_dir'],kwargs['prefix_array'] + '.'+  kwargs['ext_array'])   

    y = np.load(filearr)
    return y


def parallel_matvec(tmp_dir='tmp',prefix_matrix='A_',ext_matrix= 'npz',prefix_array='v_',ext_array='npy'):
    
    # getting mpi info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define rank master
    rank_master = 0

    # loading matrix in local mpi rank
    file = os.path.join(prefix_matrix+ str(rank) + '.' + ext_matrix)        
    Ai = sparse.load_npz(file)

    # loading vector in local mpi rank
    filearr = os.path.join(prefix_array + str(rank)  + '.' + ext_array)   
    vi = np.load(filearr)

    # local matrix and vector multiplication
    yi = Ai.dot(vi) 

    # Gather local vector to rank 0 
    global_vector_size = Ai.shape[0]
    sendbuf = yi
    recvbuf = None
    if rank==rank_master:
        recvbuf = np.empty([size, global_vector_size], dtype='d')

    comm.Gather(sendbuf, recvbuf, root=rank_master)

    # save 
    if rank == rank_master:
        y = np.sum(recvbuf.T,axis=1)
        np.save(prefix_array, y)
       

    
class  Test_Parallel(TestCase):
    def test_parallel_matvec(self):

        n = 1000
        #x = 3*np.arange(n)
        x = np.array([1.0]*n)
        mpi_size = 2
        if False:
            top = [4,-1]
            top.extend([0.0]*(n-2))
            A = linalg.toeplitz(top,top)
            A[n-1,0] = -1
            A[0,n-1] = -1
        else:
            A = sparse.rand(n, n, density=0.5, dtype=np.float, random_state=1)

        A = sparse.csc_matrix(A)

        print('Starting Serial Matrix Vector Multiplication .......')
        start_time = time.time()
        u_target = A.dot(x)
        elapsed_time = time.time() - start_time
        print('Serial matvec : Elapsed time : %f ' %elapsed_time)

        print('Starting Parallel Matrix Vector Multiplication .......')
        start_time = time.time()
        u = matvec(A,x,n=mpi_size)
        elapsed_time = time.time() - start_time
        print('Parallel matvec : Elapsed time : %f ' %elapsed_time)

        np.testing.assert_almost_equal(u_target,u,decimal=10)
        
        try:
            shutil.rmtree('tmp')
        except:
            pass

    def test_parallel_dot(self):

        n = 1000
        #x = 3*np.arange(n)
        x = np.array([1.0]*n)
        mpi_size = 2
        if False:
            top = [4,-1]
            top.extend([0.0]*(n-2))
            A = linalg.toeplitz(top,top)
            A[n-1,0] = -1
            A[0,n-1] = -1
        else:
            A = sparse.rand(n, n, density=0.5, dtype=np.float, random_state=1)

        A = sparse.csc_matrix(A)

        print('Starting Serial dot Multiplication .......')
        start_time = time.time()
        u_target = A.dot(x)
        elapsed_time = time.time() - start_time
        print('Serial matvec : Elapsed time : %f ' %elapsed_time)

        print('Starting Parallel Matrix dot Multiplication 1.......')
        start_time = time.time()
        parallelA = ParallelMatrix(A,mpi_size)
        u = parallelA.dot(x)
        elapsed_time = time.time() - start_time
        print('1 Parallel dot : Elapsed time : %f ' %elapsed_time)

        print('Starting Parallel Matrix dot Multiplication 2.......')
        start_time = time.time()
        u = parallelA.dot(x)
        elapsed_time = time.time() - start_time
        print('2 Parallel dot : Elapsed time : %f ' %elapsed_time)

        np.testing.assert_almost_equal(u_target,u,decimal=10)



if __name__=='__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size==1:

        main()


    else:
        print('Starting MPI parallel mode!')
    
        logging.basicConfig(level=logging.INFO,filename='rank_' + str(rank) + '.log')
        
        getattr_mpi_attributes(sys.argv)
