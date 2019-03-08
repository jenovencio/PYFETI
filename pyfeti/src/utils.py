import collections
import dill as pickle
import pandas as pd
from unittest import TestCase, main
from pandas.util.testing import assert_frame_equal 
from scipy import sparse
import numpy as np


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
    def __init__(self,selection_dict,id_map_df):
        ''' the selection dict contain labels as key and 
        dofs as values. The idea is to provide a class
        which can apply permutation in matrix and also global to local map
        
        parameters 
            selection_dict : Ordered dict
                dict with string and dofs
        
        '''
        self.selection_dict = selection_dict
        self.id_map_df = id_map_df
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
            for value in dof_list:
                self.local_to_global_dof_dict[local_dof_counter] = value
                self.global_to_local_dof_dict[value] = local_dof_counter
                local_dof_counter += 1
        
        
        self.P = self.create_permutation_matrix(self.local_indexes)
        self.ndof = max(self.P.shape)
    
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
        
    def create_block_matrix(self,M):
        ''' This function create block matrix with string
        which is useful for applying boundary conditions
        '''
        block_matrix = {}
        for key1, dofs_1 in self.selection_dict.items():
            for key2, dofs_2 in self.selection_dict.items():
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
        
        if return_reduced_selection:
            return sparse.vstack(M_rows).tocsc(), self.reduced_selector 
        else:
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
        B = sparse.csc_matrix((len(local_id), self.ndof), dtype=np.int8)
        B[np.arange(len(local_id)), local_id ] = 1
        return B

class MapDofs():
    def __init__(self,map_dofs):
        '''
        map_dofs as pandas dataframe
        '''
        self.map_dofs = map_dofs
   
    def get_global_dof_row_index(self,global_dof):
        return list(self.map_dofs[self.map_dofs['Global_dof_id']==global_dof].index.values.astype(int))
    
    def row2local_dof(self,row_id):
        return self.map_dofs['Local_dof_id'].ix[row_id]
        
    def row2domain_id(self,row_id):
        return self.map_dofs['Domain_id'].ix[row_id]
    
    def global_dofs_length(self):
        return max(self.map_dofs['Global_dof_id']) + 1
    
    def local_dofs_length(self,domain_id=None):
        if domain_id is None:
            return len(self.map_dofs['Local_dof_id'])
        else:
            return len(self.map_dofs[self.map_dofs['Domain_id']==domain_id]['Domain_id'])
          
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
        return list(self.map_dofs[self.map_dofs['Domain_id']==domain_id].index.values.astype(int))
    
    def local_dof(self,domain_id=None):
        if domain_id is None:
            return list(self.map_dofs['Local_dof_id'])
        else:
            return self.map_dofs[self.map_dofs['Domain_id']==domain_id]['Local_dof_id']
         
    def global_dofs(self,domain_id=None):
        if domain_id is None:
            return list(self.map_dofs['Global_dof_id'])
        else:
            return list(self.map_dofs[self.map_dofs['Domain_id']==domain_id]['Global_dof_id'])
        
    @property
    def domain_ids(self):
        return list(self.map_dofs['Domain_id'])
    
    @property
    def get_local_map_dict(self):
        domain_ids = set(self.domain_ids)
        local_map_dict = {}
        for domain_id in domain_ids:
            local_map_dict[domain_id] = self.local_dof(domain_id)
        
        return local_map_dict
        
        
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
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


# Alias for Backward compatibility
Get_dofs = DofManager
OrderedDict = collections.OrderedDict


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



if __name__ == '__main__':
    
    #main()  
    testobj = Test_Utils()
    #testobj.test_dict2dfmap()
    #testobj.test_SelectionOperator_remove_duplicate_dofs()
    testobj.test_SelectionOperator_build_B()