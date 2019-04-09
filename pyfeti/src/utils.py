import collections
import dill as pickle
import pandas as pd
from unittest import TestCase, main
from pandas.util.testing import assert_frame_equal 
from scipy import sparse
import numpy as np
import os, sys, copy, time, shutil, logging, subprocess


amfe2gmsh = {}
amfe2gmsh['Quad4'] = '3'
amfe2gmsh['straight_line'] = '1'
amfe2gmsh['node_point'] = '15'


# geting path of MPI executable
mpi_exec = 'mpiexec'
try:
    mpi_path = os.environ['MPIDIR']
    mpi_exec = os.path.join(mpi_path, mpi_exec).replace('"','')
except:
    print("Warning! Using mpiexec in global path")
    mpi_path = None
    

try:
    python_path = os.environ['PYTHON_ENV']
    python_exec = os.path.join(python_path,'python').replace('"','')
except:
    print("Warning! Using python in global path")
    python_path = None
    python_exec = 'python'

def get_mpi_exec():
    return mpi_exec

def get_python_exec():
    return python_exec

def get_platform():
    # getting amfe folder
    if sys.platform[:3]=='win':
        return 'Windows'

    elif sys.platform[:3]=='lin':
        return 'Linux'
    else :
        raise('Plataform %s is not supported  ' %sys.platform)

class MPILauncher():
    def __init__(self,python_file,mpi_size,**kwargs):

        self.python_file = python_file
        self.mpi_size = mpi_size #
        self.log = True
        self.kwargs = kwargs
        if 'tmp_folder' in self.kwargs :
            self.tmp_folder = self.kwargs['tmp_folder']
        else:
            self.tmp_folder = 'tmp'

    def run(self):

        platform = get_platform()
        if platform=='Windows':
            self.run_windowns()
        elif platform=='Linux':
            self.run_linux()

    def run_linux(self):
        run_file_path = 'run_mpi.sh'
        print('Not tested')

    def create_command_string(self,python_file,mpi_size,**kwargs):

        command = '"' + mpi_exec + '" -l -n ' + str(self.mpi_size) + ' "' + python_exec + '"  "' + \
                python_file + '"'
         
        for key, value in self.kwargs.items():
            command += '  "' + str(key) + '=' + str(value) +  '" '
        return command

    def run_windowns(self):

        run_file_path = 'run_mpi.bat'
        logging.info('######################################################################')
        logging.info('###################### SOLVER INFO ###################################')
        logging.info('MPI exec path = %s' %mpi_exec )
        logging.info('Python exec path = %s' %python_exec )

        
        command = self.create_command_string(self.python_file,self.mpi_size,**self.kwargs)

        # export results to a log file called amfeti_solver.log
        if self.log:
            command += '>mpi.log'
        
       

        # writing bat file with the command line
        local_folder = os.getcwd()
        os.chdir(self.tmp_folder)
        run_file = open(run_file_path,'w')
        run_file.write(command)
        run_file.close()

        logging.info('Run directory = %s' %os.getcwd())
        logging.info('######################################################################')

        # executing bat file
        try:    
            subprocess.call(run_file_path)
            os.chdir(local_folder)
            
        except:
            os.chdir(local_folder)
            logging.error('Error during the simulation.')
            return None

    def remove_folder(self):
        try:
            shutil.rmtree(self.tmp_folder)
        except:
            print('Could not remove the folder = %s' %(self.tmp_folder))



def getattr_mpi_attributes(system_argument):
    ''' This function call a function which supports mpi4py

    Parameters
        system_argument : sys.args
            sys.args must have the follow format
            "method=python_func" "arg1=value1" "arg2=value2"

            such that the python function that supports mpi has the
            following calling interface:

            python_func(arg1=value1,arg2=value2)

            the "method=python_func" is a mandatory argument in the command line


        Return:
            None
    '''

    args = []
    for s in system_argument:
        args.append(s)    
        
    mpi_kwargs = {}
    for arg in args[1:]:
        try:
            var, value = arg.split('=')
            try:
                mpi_kwargs[var] = int(value)
            except:
                mpi_kwargs[var] = value
        except:
            print('Commnad line argument noy understood, arg = %s cannot be splited in variable name + value' %arg)


        
    if 'module' not in mpi_kwargs:
        raise AttributeError('module must be passed to mpi call, e.g. module=MPIlinalg')

    if 'method' not in mpi_kwargs:
        raise AttributeError('method must be passed to mpi call, e.g. method=parallel_matvec')
    

    module = __import__(mpi_kwargs['module'])
    method_to_call = getattr(module, mpi_kwargs['method'])

    # removing method to the kwargs
    del mpi_kwargs['method']
    del mpi_kwargs['module']

    # calling the parallel mpi function
    method_to_call(**mpi_kwargs)

    return None


class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


class DofManager():
    def __init__(self,id_map_df ):
        ''' 
        paramentes:
        id_map_df: pandas.DataFrame
            id_map_df is a pandas dataframe where columns
            represent directions and rows represent nodes id
    
        '''
        self.id_map_df = id_map_df
        
    def get(self,global_node_list, direction ='xyz'):
        ''' get dofs given a submesh and a global id_matrix
        
        parameters:
            global_node_list : list
                List with the global node IDs
            direction : str
                direction to consider 'xyz'

        # return 
            # dir_dofs : list
                # list with Dirichlet dofs
        '''
     
        dir_dofs = []
        for dir in ['x','y','z']:
            if dir in direction:
                try:
                    dir_dofs.extend(list(self.id_map_df[dir][global_node_list]))
                except:
                    print('Not possible to local dof based on the given node list')
                
        dir_dofs.sort()
        return dir_dofs

 
class SelectionOperator():
    def __init__(self,selection_dict,id_map_df,remove_duplicated = True):
        ''' the selection dict contain labels as key and 
        dofs as values. The idea is to provide a class
        which can apply permutation in matrix and also global to local map
        
        parameters 
            selection_dict : Ordered dict
                dict with string and dofs
        
        '''
        self.selection_dict = copy.deepcopy(selection_dict)
        self.id_map_df = id_map_df
        if remove_duplicated:
            self._remove_duplicate_dofs()

        self.all_keys_set = OrderedSet(self.selection_dict.keys())
        self.removed_keys = OrderedSet()
        self.red_dof_dict = None
        self.local_indexes = []
        self.bounds = {}
        self.length = {}
        self.local_to_global_dof_dict = {} 
        self.global_to_local_dof_dict = {} 
        self.global_id_map_df = id_map_df
        

        count = 0
        local_dof_counter = 0
        for key, dof_list in selection_dict.items():
            self.local_indexes.extend(dof_list)
            length = len(self.local_indexes)
            self.length[key] = len(dof_list)
            self.bounds[key] = [count,length]
            count += length
            for value in list(dof_list):
                self.local_to_global_dof_dict[local_dof_counter] = value
                self.global_to_local_dof_dict[value] = local_dof_counter
                local_dof_counter += 1
        
        
        self.P = self.create_permutation_matrix(self.local_indexes)
        self.ndof = max(self.id_map_df.max()) + 1
    
    @property
    def list_of_all_dofs(self):
        return list(range(max(self.id_map_df.max())+1))

    def _remove_duplicate_dofs(self):
        
        list_of_all_dofs = self.list_of_all_dofs
        subset_list = []
        for key, value in self.selection_dict.items():
            
            dif_set = value - OrderedSet(subset_list)
            self.selection_dict.update({key : dif_set})
            subset_list.extend(list(dif_set ))

        if 'internal' not in self.selection_dict:
            self.selection_dict.update({'internal' : OrderedSet(list_of_all_dofs) - OrderedSet(subset_list)})

    def nodes_to_local_dofs(self):
        pass
    
    def create_permutation_matrix(self,local_indexes):
        ''' create a Permutation matrix based on local id
        
        '''
        ndof = len(local_indexes)
        P = sparse.lil_matrix((ndof, ndof), dtype=np.int8)
        P[local_indexes, np.arange(ndof)] = 1
        return P.T.tocsc()
        
    def create_block_matrix(self,M,block_keys=None):
        ''' This function create block matrix with string
        which is useful for applying boundary conditions
        
        Parameters :
            M : np.array or sparse matrix
                matrix to be decomposed
            block_keys : list, default : None
                list of keys to create the block matrix
                if None all keys in the self.selection_dict will be use
        '''
        
        if block_keys is None:
            block_keys = list(self.selection_dict.keys())
           
        block_matrix = {}           
        for key1 in block_keys:
            dofs_1 = self.selection_dict[key1]
            for key2 in block_keys:
                dofs_2 = self.selection_dict[key2]
                block_matrix[key1,key2] = M[np.ix_(list(dofs_1), list(dofs_2))]
        
        return block_matrix
        
    def create_block_vector(self,f):
        block_vector = {}
        for key1, dofs_1 in self.selection_dict.items():
            block_vector[key1] = f[dofs_1]
        
        return block_vector
    
    def assemble_matrix(self,M,list_of_strings,return_reduced_selection=False):
        ''' This method assemble a matrix based on the list of string
        useful for ordering the matrix according to the block string matrix
        paramenter:
            M : np.array
                matrix to be reordered
            list of strings : list
                list with a sequence of string which gives the 
                order of the degrees of freedom associated with M11
            return_reduced_selection : Boolean
                return a new SelctionOperator for the recuded system
            return a ordered Matrix
            
            ex. 
                M_block = s.create_block_matrix(M)
                M_row1 = sparse.hstack((M_block['l','l'],M_block['l','h'],M_block['l','i']))
                M_row2 = sparse.hstack((M_block['h','l'],M_block['h','h'],M_block['h','i']))
                M_row3 = sparse.hstack((M_block['i','l'],M_block['i','h'],M_block['i','i']))
                M_sector = sparse.vstack((M_row1,M_row2,M_row3)).tocsc()
            
            
        '''
        

        self.create_reduced_selector(list_of_strings)
        
        M_block = self.create_block_matrix(M)
        
        M = self.assemble_block_matrix(M_block,list_of_strings)
        
        if return_reduced_selection:
            return M, self.reduced_selector 
        else:
            return M
    
    def assemble_block_matrix(self,M_block,list_of_strings):
        M_rows = []
        for s_i in list_of_strings:
            M_row_j_list = [] 
            for s_j in list_of_strings:
                Mij = M_block[s_i,s_j]
                if sparse.issparse(Mij):
                    M_row_j_list.append(Mij)
                else:
                    M_row_j_list.append(sparse.csr_matrix(Mij))
            M_rows.append(sparse.hstack(M_row_j_list))
        
        return sparse.vstack(M_rows).tocsc()
        
    def assemble_vector(self,f,list_of_strings):
        ''' This method assemble a vector based on the list of string
        useful for ordering the matrix according to the block string matrix
        paramenter:
            M : np.array
                1-d array to be reordered
            
            list of strings : list
                list with a sequence of string which gives the 
                order of the degrees of freedom associated with M11
            
            return a ordered Matrix
            
            
        '''
        
        f_block = self.create_block_vector(f)
        
        f_rows = np.array([])
        for s_i in list_of_strings:
            f_rows = np.append(f_rows, f_block[s_i])
        
        return f_rows
        
    def select_block(self,M,rows,columns):
        pass
    
    def create_reduced_selector(self,list_of_strings):
        
        self.removed_keys =  self.all_keys_set - list_of_strings # copy list with all keys
        self.red_dof_dict = collections.OrderedDict()
        init_dof = 0
        for key in list_of_strings:
            last_dof = init_dof + len(self.selection_dict[key])
            self.red_dof_dict[key] = OrderedSet(np.arange(init_dof,last_dof))
            init_dof = last_dof
        
        self.reduced_selector = SelectionOperator(self.red_dof_dict,self.global_id_map_df)

    def build_B(self,label):
        ''' Build Boolean selection operator
        
        '''
        
        local_id = list(self.selection_dict[label])
        B = sparse.lil_matrix((len(local_id), self.ndof), dtype=np.int8)
        B[np.arange(len(local_id)), local_id ] = 1
        return B.tocsr()
    
    def get_union_of_dofs(self,key_list):
        ''' This function returns the union of the 
        node given by the key_list
        
        Parameters:
            key_list : list
                list of the keys of the self.selection_dict
        
        Returns:
            set of the union off dofs
            
        '''
        node_set = set()
        for key in key_list:
            node_set.update(self.selection_dict[key])
            
        return node_set
    
    def get_difference_set(self,key_list_1,key_list_2):
        ''' This function returns the diffence of the union 
        of the two sets key_list_1 and key_list_2
        
        Parameters:
            key_list_1 : list
                list of the keys of the self.selection_dict to perform a union
            key_list_2 : list
                list of the keys of the self.selection_dict to perform a union
                
        Returns 
             the difference set of dofs
        '''
        union_1 =  self.get_union_of_dofs(key_list_1)
        union_2 =  self.get_union_of_dofs(key_list_2)
        return union_1 - union_2
    
    def get_complementary_set(self,key_list):
        ''' This function return the complementary set
        of the union of the key_list minus all dofs
        '''
        all_set = set(self.list_of_all_dofs)
        given_set = self.get_union_of_dofs(key_list)
        return all_set - given_set
    
    def add_difference_set_into_dict(self,key_list_1,key_list_2,key_id,overwrite=False):
        ''' This function add to the self.selection_dict 
        the difference set, defined as the union of the set
        given by the key_list minus the list of of dofs
        
        Parameters:
            key_list_1 : list
                list of the keys of the self.selection_dict to perform a union
            key_list_2 : list
                list of the keys of the self.selection_dict to perform a union
            key_id : int or string
                key for the addition of the new dict
            
        Return 
            None
            
        '''
        if (key_id not in self.selection_dict) or overwrite:
            self.selection_dict[key_id] = self.get_difference_set(key_list_1,key_list_2)
        else:
            raise Exception('Given key_id is already is selection dictionary!')
            


class MapDofs():
    def __init__(self,map_dofs,primal_tag='Global_dof_id',local_tag='Local_dof_id',domain_tag='Domain_id'):
        '''
        map_dofs as pandas dataframe
        '''
        self.map_dofs = map_dofs
        self.primal_tag = primal_tag
        self.local_tag = local_tag
        self.domain_tag = domain_tag
   
    def get_global_dof_row_index(self,global_dof):
        return list(self.map_dofs[self.map_dofs[self.primal_tag]==global_dof].index.values.astype(int))
    
    def row2local_dof(self,row_id):
        return self.map_dofs[self.local_tag].iloc[row_id]
        
    def row2domain_id(self,row_id):
        return self.map_dofs[self.domain_tag].iloc[row_id]
    
    def global_dofs_length(self):
        return max(self.map_dofs[self.primal_tag]) + 1
    
    def local_dofs_length(self,domain_id=None):
        if domain_id is None:
            return len(self.map_dofs[self.local_tag])
        else:
            return len(self.map_dofs[self.map_dofs[self.domain_tag]==domain_id][self.domain_tag])
          
    def global2local_dof(self,global_dof):
        return (list(map(self.row2local_dof,self.get_global_dof_row_index(global_dof))), 
                list(map(self.row2domain_id,self.get_global_dof_row_index(global_dof))))
                
    def get_local_dof(self, global_dof, domain_id):
        local_dofs_list, domain_id_list = self.global2local_dof(global_dof)
        if domain_id in  domain_id_list:
            local_dofs = local_dofs_list[domain_id_list.index(domain_id)]
            return local_dofs
        else:
            return None
    
    def get_domain_rows(self,domain_id):
        return list(self.map_dofs[self.map_dofs[self.domain_tag]==domain_id].index.values.astype(int))
    
    def local_dof(self,domain_id=None):
        if domain_id is None:
            return list(self.map_dofs[self.local_tag])
        else:
            return self.map_dofs[self.map_dofs[self.domain_tag]==domain_id][self.local_tag]
         
    def global_dofs(self,domain_id=None):
        if domain_id is None:
            return list(self.map_dofs[self.primal_tag])
        else:
            return list(self.map_dofs[self.map_dofs[self.domain_tag]==domain_id][self.primal_tag])
        
    @property
    def domain_ids(self):
        return list(self.map_dofs[self.domain_tag])
    
    @property
    def get_local_map_dict(self):
        domain_ids = set(self.domain_ids)
        local_map_dict = {}
        for domain_id in domain_ids:
            local_map_dict[domain_id] = self.local_dof(domain_id)
        
        return local_map_dict
        
def save_object(obj, filename,tries=2,sleep_delay=3 ):
    filename = r"{}".format(filename)
    for i in range(tries):
        try:
            with open(filename, 'wb') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
                break
        except:
            time.sleep(sleep_delay)
            continue

def load_object(filename,tries=3,sleep_delay=5):
    ''' load a pickle object

    parameters:
        filename : string
            path of the file to be open
        tries : int
            number of tries before exit
        sleep_delay : int
            seconds to wait until next try

    Returns
        pickle object


    '''
    filename = r"{}".format(filename)
    obj = None
    for i in range(tries):
        try:
            with open(filename, 'rb') as input:
                obj = pickle.load(input)
            break
        except:
            time.sleep(sleep_delay)
            continue
    if obj is None:
        print('Could not find the file path %s ' %filename)
        raise FileNotFoundError
    return obj

def dict2dfmap(id_dict,column_labels=None):
    ''' This function converts a dictionary id_matrix 'id_dict' to
    a pandas dataframe id_matrix 'id_df'.
    Where the id_matrix has node id as keys and a list of dofs as values
    e.g. id_dict[0] = [0,1,2]
         id_dict[1] = [3,4,5]

    Parameters
        id_dict : dict
            dictionary that maps node ids to dofs

        column_labels : list , Default = None
            list with the strings of the columns dofs e.g. ['x','y','z']
            if None, the column_labels will be guess by the size of the list of the node id = 0

    Return : pandas.Dataframe

        e.g.   |  x | y | z
             0 | 0  | 1 | 2
             1 | 3  | 4 | 5
        
    '''
    if column_labels is None:
        if len(id_dict[0])==2:
            column_labels = ['x','y']
        elif len(id_dict[0])==3:
            column_labels = ['x','y','z']
        else: 
            raise('Error! Please, provide columns_labels to create a proper dataframe map.')
    


    return pd.DataFrame.from_dict(id_dict, columns=column_labels,orient='index')

def create_selection_operator(node_df,element_df,tag='phys_group',remove_duplicated = False, unique_id = -1):
    ''' This function creates a SelectionOperator object based on mesh information.
    
    Parameters:
    
    node_df : pandas.dataframe
        dataframe containing the map between node id and dofs
        
    element_df : pandas.dataframe 
        dataframe containing elem id in rows and columns with connectivity
        and tags which have elements in groups
        
    tag : string
        string group to create the selection operator
        
    remove_duplicated : Boolean default = False
        Boolean operator that forces a unique set of dofs per index in the SelectionOperator.
        
    unique_id : int or string
        unique id for the dofs which do not belong to any group tag
        
    Returns:
        A SelctionOperator obj
    '''
    
    if 'connectivity' not in element_df:
        raise(Exception)
    
    
    
    get_dof_from_node = lambda node_id : list(node_df.iloc[node_id])
    get_connectivity_from_elem_id = lambda elem_id : list(element_df['connectivity'].iloc[elem_id])
    get_dof_list_from_elem_id = lambda elem_id : list(map(get_dof_from_node,get_connectivity_from_elem_id(elem_id)))
    get_elements_from_group_id = lambda group_id : list(element_df.loc[element_df[tag] == group_id].index)
    get_dof_list_from_group_id = lambda group_id : np.array(list(map(get_dof_list_from_elem_id,get_elements_from_group_id(group_id)))).flatten()
    
    
    group_set = set(element_df[tag])
    if unique_id in group_set:
        print('Group tag uses the unique id %i, please ')
        raise(Exception)
    
    all_dofs = set(node_df.values.flatten())
    dof_dict = {}
    dofs = set()
    for group_id in set(element_df[tag]):
        group_dofs = get_dof_list_from_group_id(group_id)
        dof_dict[group_id] = set(group_dofs)
        dofs.update(group_dofs)

    dof_dict[unique_id] = list(all_dofs - dofs)   

    return SelectionOperator(dof_dict,node_df,remove_duplicated = remove_duplicated)

class Log():
    def __init__(self,filename):
        self.filename = filename
        self.text_list = []
        
    def append(self,string):
        self.text_list.append(string)
        
    def save(self):
        with open(self.filename,'w') as file:
            for line in self.text_list:
                file.write(line + '\n')

# Alias for Backward compatibility
Get_dofs = DofManager
OrderedDict = collections.OrderedDict

def pyfeti_dir(filename=''):
    '''
    Return the absolute path of the filename given relative path
    directory.

    Parameters
    ----------
    filename : string, optional
        relative path to something inside the amfe directory.

    Returns
    -------
    dir : string
        string of the filename inside the pyfeti-directory. Default value is ''.

    '''
    pyfeti_abs_path = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(pyfeti_abs_path, filename.lstrip('/'))



class DomainCreator():
    def __init__(self,width=10,high=10,x_divisions=11,y_divisions=11,domain_id=1):
        self.width = width
        self.high = high
        self.x_divisions = x_divisions
        self.y_divisions = y_divisions
        self.domain_id = domain_id
        self.start_x = 0.0
        self.start_y = 0.0
        self.elem_type = 'Quad4'
        self.gmsh_type = 1
        self.elem_num = 0
        self.tag_dict = {}

    def node_map(self):
        a = self.x_divisions 
        b = 1
        c = 1
        return lambda tup : a*tup[0] + b*tup[1] + c

    def build_nodes(self):
        delta_x = self.width/(self.x_divisions-1)
        delta_y = self.high/(self.y_divisions-1)
        x0 = self.start_x
        y0 = self.start_y
        nodes_dict = {}
        for i in range(self.y_divisions):
            for j in range(self.x_divisions):
                nodes_dict[i,j] = [x0 + j*delta_x , y0 + i*delta_y, 0.0]
    
        return nodes_dict

    def build_elements(self):
        ''' This function build linear and quad elements based on instance
        parameters
        '''

        self.elem_num = 0

        elem_dict = {}
        node_dict = self.create_node_points()
        linear_elem_dict = self.create_linear_elem()
        quad_elem_dict = self.create_quad_elem()
        
        elem_dict.update({'node_point' : node_dict}) 
        elem_dict.update({'straight_line' : linear_elem_dict}) 
        elem_dict.update({'Quad4' : quad_elem_dict})

        return elem_dict

    def create_node_points(self):
        ''' This function create dict node points 
        
        Returns 
            dict [tag_name] = elem_dict
        '''
        node_map = self.node_map()
        node_elem_dict = {'corner_BL' : {0 : [node_map((0,0))]} ,
                          'corner_BR' : {1 : [node_map((0,self.x_divisions-1))]} ,
                          'corner_TL' : {2 : [node_map((self.y_divisions-1,0))]} ,
                          'corner_TR' : {3 : [node_map((self.y_divisions-1,self.x_divisions-1))]}}
        self.elem_num += 4
        # update self.tag_dict with elem_dict information
        self.tag_dict.update({'corner_BL' : '6' , 'corner_BR' : '7' , 'corner_TL' : '8' , 'corner_TR' : '9'}) 

        return node_elem_dict

    def create_linear_elem(self):
        ''' This function create dict straing linear elements 
        
        Returns 
            dict [tag_name] = elem_dict
        '''
        node_map = self.node_map()
        
        linear_elem_dict = {'bottom' : {} , 'right' : {} , 'top' : {} , 'left' : {}}
        delta_dict = {'bottom' : (0,1) , 'right' : (1,0) , 
                      'top' : (0,-1)  , 'left' : (-1,0)}
        division_map = {'bottom' : self.x_divisions , 'right' : self.y_divisions , 
                      'top' : self.x_divisions  , 'left' : self.y_divisions }
        node_index_pair = (0,0)
        for key, linear_dict in linear_elem_dict.items():
            for node_mult in range(division_map[key]-1):
                next_node_index_pair = tuple( np.array(node_index_pair) +  np.array(delta_dict[key]))
                linear_dict[node_mult] = list(map(node_map, [node_index_pair,next_node_index_pair]))
                node_index_pair = next_node_index_pair
                self.elem_num += 1
        # update self.tag_dict with elem_dict information
        self.tag_dict.update({'bottom' : '4' , 'right' : '2' , 'top' : '5' , 'left' : '1'}) 

        return linear_elem_dict

    def create_quad_elem(self,tag_name='domain'):
        ''' This function create dict Quad4 elements
        with the key given by the tag_name
        
        Parameters
            tag_name : string
                key for the element dictionaty

        Returns 
            dict [tag_name] = elem_dict
        '''
        node_map = self.node_map()
        quad_elem_nodes = lambda I,J : [(I,J),(I,J+1),(I+1,J+1),(I+1,J)]

        quad_elem_dict = {tag_name : {}}
        count = 0
        for elem_id_j in range(self.y_divisions - 1):
            for elem_id_i in range(self.x_divisions-1):
                quad_elem_dict[tag_name][count] =  list(map(node_map,quad_elem_nodes(elem_id_j,elem_id_i)))
                count+=1
        self.elem_num += count 
        # update self.tag_dict with elem_dict information
        self.tag_dict.update({'domain' : '3'}) 

        return quad_elem_dict

    def save_gmsh_file(self,filename, format = '2.2 0 8' ):
        
        tag_dict = self.tag_dict

        nodes_dict = self.build_nodes()
        elem_dict = self.build_elements()
        
        format_string = self.create_gmsh_format_string(format)
        nodes_string = self.create_gmsh_nodes_string(nodes_dict)
        phys_string = self.create_phys_string(tag_dict)
        elem_string = self.create_gmsh_elem_string(elem_dict,tag_dict)

        mesh_string = [format_string,phys_string,nodes_string,elem_string]
        with open(filename,'w') as f:
            for s in mesh_string:
                f.write(s)

    def create_gmsh_format_string(self,format):
        tag_format_start   = "$MeshFormat"
        tag_format_end     = "$EndMeshFormat"
        format_string_list = [tag_format_start,
                              format,
                              tag_format_end]

        format_string = ''
        for s in format_string_list:
            format_string += s + '\n'
        return format_string

    def create_phys_string(self,tag_dict):

        tag_phys_start   = "$PhysicalNames"
        tag_phys_end     = "$EndPhysicalNames"

        phys_string = tag_phys_start + '\n'
        phys_string += str(len(tag_dict.keys())) + '\n'
        phys_count = 1
        for key, value in tag_dict.items():
            phys_string += ' '.join([str(phys_count),value,'"' + key + '"']) + '\n'
            phys_count += 1

        return phys_string

    def create_gmsh_nodes_string(self,nodes_dict):

        tag_nodes_start    = "$Nodes"
        tag_nodes_end      = "$EndNodes\n"

        nodes_string = tag_nodes_start + '\n'
        nodes_string += str(len(nodes_dict.keys())) + '\n'
        count = 1
        for key,item in nodes_dict.items():
            nodes_string +=  str(count) + ' ' +  ' '.join(list(map(str,item))) + '\n'
            count += 1
    
        nodes_string += tag_nodes_end
    
        return nodes_string

    def create_gmsh_elem_string(self,elem_dict,tag_dict={}):
       
        tag_elements_start = "$Elements"
        tag_elements_end   = "$EndElements"
        num_of_tags = 4
        partition_num = 1
        partition_id = 1

        elem_string = ''.join(tag_elements_start)
        elem_string += '\n' + str(self.elem_num) + '\n'
        phys_tag_count=1 
        elem_count = 1
        for amfe_elem_tag, item in elem_dict.items():
            gmsh_elem_tag = amfe2gmsh[ amfe_elem_tag]
            for phys_tag, elem_dict in item.items():
                if tag_dict:
                    phys_tag = tag_dict[phys_tag]
                else:
                    phys_tag = str(phys_tag_count)
                geo_tag = phys_tag
                for key, nodes in elem_dict.items():
                    row_i =  ' '.join([str(elem_count),gmsh_elem_tag,str(num_of_tags),phys_tag,geo_tag,str(partition_num),str(partition_id)])
                    elem_string += row_i + ' ' + ' '.join(list(map(str,nodes))) + '\n'
                    elem_count +=1
            phys_tag_count += 1
        elem_string += tag_elements_end
        return elem_string

class  Test_Utils(TestCase):
    def test_OrderedSet(self):
        s = OrderedSet('abracadaba')
        t = OrderedSet('simsalabim')
        print(s | t)
        print(s & t)
        print(s - t)

    def test_dict2dfmap(self):
        target_df = pd.DataFrame(data={'x': [0, 2], 'y': [1, 3]})

        id_dict = {}
        id_dict[0] = [0,1]
        id_dict[1] = [2,3]
        id_df = dict2dfmap(id_dict)

        assert_frame_equal(id_df, target_df)

    def test_SelectionOperator_remove_duplicate_dofs(self):
        id_df = pd.DataFrame(data={'x': [0, 2, 4, 6], 'y': [1, 3, 5, 7]})
        group_dict = OrderedDict()
        group_dict[1] = OrderedSet([0,1,2])
        group_dict[2] = OrderedSet([2,4])
        group_dict[3] = OrderedSet([0,3])

        s = SelectionOperator( group_dict, id_df)
        self.assertEqual( list(s.selection_dict[2])[0],4)
        self.assertEqual( list(s.selection_dict[3])[0],3)
        self.assertEqual( list(s.selection_dict['internal']),[5,6,7])


    def test_SelectionOperator_build_B(self):
        id_df = pd.DataFrame(data={'x': [0, 2, 4, 6], 'y': [1, 3, 5, 7]})
        group_dict = OrderedDict()
        group_dict[1] = OrderedSet([0,1,2])
        group_dict[2] = OrderedSet([2,4])
        group_dict[3] = OrderedSet([0,3])

        s = SelectionOperator( group_dict, id_df)
        s.build_B(1)

    def test_DomainCreator(self):
        creator_obj  = DomainCreator(x_divisions=4,y_divisions=3)
        creator_obj.build_elements()
        try:
            os.mkdir('meshes')
        except:
            pass

        mesh_path = r'meshes\mesh1.msh'
        creator_obj.save_gmsh_file(mesh_path)
        shutil.rmtree('meshes')

    def test_mpi_launcher(self):
        python_script = """from mpi4py import MPI
import sys\ncomm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('UnitTest from rank = %i' %rank)
for s in sys.argv:
    print(s) """

        tmp_folder = 'tmp'
        try:
            os.mkdir(tmp_folder)
        except:
            pass

        dummy_script = r'tmp\test.py' 
        dummy_file = 'test.py'
        with open(dummy_script,'w') as f:
            f.write(python_script)

        mpi_launcher = MPILauncher(dummy_file,2,solver='PCG',pseudoinverse='SuperLU')
        mpi_launcher.run()

        target_string = '''[1]UnitTest from rank = 1
[1]test.py
[1]solver=PCG
[1]pseudoinverse=SuperLU
[0]UnitTest from rank = 0
[0]test.py
[0]solver=PCG
[0]pseudoinverse=SuperLU'''

        string_list = target_string.split('\n')
        with open(r'tmp\mpi.log','r') as f:
            txt_string = f.readlines()

            
        #checking only the first line and last line
        self.assertEqual( txt_string[0][3:-3],string_list[0][3:-2])
        self.assertEqual( txt_string[-1][3:-3],string_list[-1][3:-2])

        try:
            shutil.rmtree(tmp_folder)
        except:
            print('Could not remove the folder %s' %(tmp_folder))
            pass

if __name__ == '__main__':
    
    main()  
    #testobj = Test_Utils()
    #testobj.test_dict2dfmap()
    #testobj.test_SelectionOperator_remove_duplicate_dofs()
    #testobj.test_SelectionOperator_build_B()
    #testobj.test_DomainCreator()
    #testobj.test_mpi_launcher()