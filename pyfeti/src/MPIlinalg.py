import numpy as np
from scipy import sparse
from scipy import linalg
import os, sys, shutil
from unittest import TestCase, main
import logging
import time
from mpi4py import MPI
from pyfeti.src.utils import MPILauncher, getattr_mpi_attributes, pyfeti_dir
from scipy.sparse.linalg import LinearOperator
from pyfeti.src.linalg import RetangularLinearOperator, vector2localdict, array2localdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
    logging.debug('Init exchange_info')

    #init neighbor variable
    var_nei = None
    if nei_id>0:
        #checking data type
        if isnumpy:
            
                # sending message to neighbors
                comm.Send(local_var, dest = nei_id-1)
                # receiving messages from neighbors
                var_nei = np.empty(local_var.shape)
                comm.Recv(var_nei,source=nei_id-1)
        else:
            
            var_nei  = comm.sendrecv(local_var,dest=nei_id-1,source=nei_id-1)

    logging.debug('End exchange_info')
    return var_nei

def exchange_global_dict(local_dict,local_id,partitions_list):
    
    logging.debug('Init exchange_global_dict')
    logging.debug(('local_dict =' ,local_dict))
    for global_id in partitions_list:
        if global_id!=local_id:
            nei_dict =  exchange_info(local_dict,local_id,global_id)
            if nei_dict:
                local_dict.update(nei_dict)

    logging.debug('End exchange_global_dict')
    return local_dict
    
def exchange_global_array(local_array,local_id,partitions_list=None):
    ''' This function receives an array and send it to all mpi ranks
    which is equivalente to a broadcast, and create a dict of arrays

    parameters: 
        local_array : np.array
            array to send to all mpi ranks
        local_id : int
            id of the local domain, rank + 1
        partitions_list : list Default : None
            list of all domains

    returns:
        local_dict : dict
            dict with domain keys and arrays as values

    '''
    
    if partitions_list is None:
        size = comm.Get_size()
        partitions_list = list(range(1,size+1))

    local_dict = {}
    local_dict[local_id] = local_array
    for global_id in partitions_list:
        if global_id!=local_id:
            nei_array =  exchange_info(local_array,local_id,global_id,isnumpy=False)
            local_dict[global_id] = nei_array

    return local_dict

def pardot(v,w,local_id,neighbors_id,global2local_map,partitions_list=None):
    ''' This function computes a parallel dot product v * w
    based on mpi operations

    parameters:
        v : np.array
            array to perform v.dot(w)
        w : np.array
            array to perform w.dot(v)
        local_id : int
            id of local problem
        neighbors_id : list
            list of neighbor ids
        global2local_map : callable
            function to convert array to dict of arrays
        partitions_list: list, Default=None
            list of all the mpi
        

    '''

    if partitions_list is None:
        size = comm.Get_size()
        partitions_list = list(range(1,size+1))

    partial_norm_dict = {}
    v_dict = vector2localdict(v, global2local_map)
    w_dict = vector2localdict(w, global2local_map)
    for nei_id in neighbors_id:
        if nei_id>local_id:
            key_pair = (local_id,nei_id)
        else:
            key_pair = (nei_id,local_id)

        local_v = v_dict[key_pair]
        local_w = w_dict[key_pair]

        if type(local_w) is not np.ndarray:
            logging.error('pardot method received a variable that is not a np.array!')
            return None

        # averaging among neighbors
        local_var = local_v.dot(local_w)
        nei_var = exchange_info(local_var,local_id,nei_id,isnumpy=True)
        partial_norm_dict[key_pair] = 0.5*(local_var + nei_var)

    # global exchange with scalars
    partial_norm_dict = exchange_global_dict(partial_norm_dict,local_id,partitions_list)

    v_dot_w = 0.0
    for (i_id,j_id) , item in partial_norm_dict.items():
        if j_id>=i_id:
            v_dot_w+=item
    
    #logging.info(('v_dot_w',v_dot_w))
    return v_dot_w

def get_chunks(number_of_chuncks,size):
    ''' create chuncks based on number of mpi process
    '''

    n = size
    remaining = n%number_of_chuncks
    default_size =  int((n-remaining)/number_of_chuncks)
    chunck_sizes = np.linspace(default_size,n,number_of_chuncks,dtype=np.int)
    chunck_sizes[-1] += remaining
    return chunck_sizes

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
    python_file = pyfeti_dir(os.path.join('src','MPIlinalg.py'))
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


class ParallelRetangularLinearOperator(RetangularLinearOperator):
    def __init__(self,A_dict,row_map_dict,column_map_dict,shape=(0,0),dtype=np.float,**kwargs):
        super().__init__(A_dict,row_map_dict,column_map_dict,shape,dtype)
        self.local_id = rank + 1
        self.kwargs = kwargs
        self.__dict__.update(kwargs)

    def _matvec(self,v, **kwargs):

        local_id = self.local_id
        # convert vector to dict    
        v_dict = self.vec2dict(v, **kwargs)
        a = np.zeros(self.shape[0])
        for nei_id in self.neighbors_id:
            if nei_id>=local_id:
                pair = (local_id,nei_id)
            else:
                pair = (nei_id,local_id)
            
            # Try matvec, except is the Transpose multiplication
            if (self.local_id,nei_id) in self.A_dict:
                A = self.A_dict[self.local_id,nei_id]
                try:
                    # A.dot(x) implementation
                    a[self.row_map_dict[self.local_id]] += A.dot(v_dict[pair])
                except:
                    # A.T.dot(x) implementation
                    a[self.row_map_dict[pair]] += A.T.dot(v_dict[self.local_id])
            else:
                pass

        return self._callback(a)

    def _callback(self,a):
        local_id = self.local_id
        try:       
            if isinstance(list(self.row_map_dict.keys())[0],int):
                vec2dict = array2localdict(a, self.row_map_dict)
                if self.local_id in vec2dict:
                    vec2dict = exchange_global_array(vec2dict[self.local_id],self.local_id)
                else:
                    vec2dict = exchange_global_array(np.array([0.]),self.local_id)
                a = self.dict2vec(vec2dict,a.shape[0],self.row_map_dict)
                
            else:
                
                vec2dict = array2localdict(a, self.row_map_dict)
                for nei_id in self.neighbors_id:
                    if nei_id!=local_id:
                        if nei_id>local_id:
                            pair = (local_id,nei_id)
                        else:
                            pair = (nei_id,local_id)

                        local_var = vec2dict[pair]
                        nei_var = exchange_info(local_var,rank+1,nei_id,isnumpy=True)    
                        vec2dict[pair] = local_var + nei_var
                    else:
                        pass
                
                a = self.dict2vec(vec2dict,a.shape[0],self.row_map_dict)
        except:
            pass

        return a 

    def _transpose(self):
        return ParallelRetangularLinearOperator(self.A_dict,self.column_map_dict,self.row_map_dict,
                                        shape=(self.shape[1],self.shape[0]),dtype=self.dtype,**self.kwargs)


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
    

    if size==1:

        main()


    else:
        print('Starting MPI parallel mode!')
    
        logging.basicConfig(level=logging.INFO,filename='rank_' + str(rank) + '.log')
        
        getattr_mpi_attributes(sys.argv)
