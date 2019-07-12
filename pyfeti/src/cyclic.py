from unittest import TestCase, main
import numpy as np
import copy
import pandas as pd
import scipy.sparse as sparse
import scipy.linalg as linalg
from pyfeti.src.utils import SelectionOperator, create_selection_operator, DofManager, OrderedSet, OrderedDict


def maptoglobal(modes_dict,s):
    global_dofs = s.P.shape[0]
    new_modes_dict = {}
    for i in modes_dict:
        modes = modes_dict[i]
        local_dofs, num_modes = modes.shapeDofManager
        zeros = np.zeros((global_dofs - local_dofs,num_modes))
        local_modes = np.vstack((zeros,modes))
        global_modes = s.P.T.dot(local_modes)
        new_modes_dict[i] = global_modes
    return new_modes_dict

def create_voigt_rotation_matrix(n_dofs,alpha_rad, dim=2, unit='rad', sparse_matrix = True):
    ''' This function creates voigt rotation matrix, which is a block
    rotation which can be applied to a voigt displacement vector
    ''' 
    
    if n_dofs<=0:
        raise('Error!!! None dof was select to apply rotation.')
    
    if unit[0:3]=='deg':
        rotation = np.deg2rad(rotation)
        unit = 'rad'
        
    R_i = get_unit_rotation_matrix(alpha_rad,dim)  
    
    n_blocks = int(n_dofs/dim)
    
    
    if n_blocks*dim != n_dofs:
        raise('Error!!! Rotation matrix is not matching with dimension and dofs.')
    if sparse_matrix:
        R = sparse.block_diag([R_i]*n_blocks)
    else:
        R = linalg.block_diag(*[R_i]*n_blocks)
    return R

def get_unit_rotation_matrix(alpha_rad,dim):  
    
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    
    if dim==3:
        R_i = np.array([[cos_a,-sin_a,0.0],
                        [sin_a, cos_a,0],
                        [0.0, 0.0, 1.0]])
    elif dim==2:
        R_i = np.array([[cos_a, -sin_a],
                       [sin_a, cos_a]])    
    else:
        raise('Dimension not supported')      
        
    return R_i


class Contact():
    ''' This is a class to hanble contact element

        parameters:
            master_nodes : list 
                list with nodes id

            slave_nodes: list
                list with nodes id

            nodes_coord : np.array
                array with node coordinates

            connectivity  : list
                element connectivity

    '''

    def __init__(self, master_nodes, slave_nodes, nodes_coord, connectivity=None,type = 'node2node',nodes_coord_slave=None, tol_radius = 1e-6):
        
        self.master_nodes = master_nodes 
        self.slave_nodes = slave_nodes
        self.nodes_coord = nodes_coord
        self.connectivity = connectivity
        self.contact_elem_dict = {}
        self.master_normal_dict = {}
        self.tol_radius = tol_radius

        if nodes_coord_slave is None:
            self.nodes_coord_slave = nodes_coord
        else:
            self.nodes_coord_slave = nodes_coord_slave

        if type == 'node2node':
            pass
        else:
            print('Type of contact not implemented!')
            return None
    
    def find_node_pairs(self, tol_radius = None):
        ''' find node pairs for contact given two submeshs

       
            tol_radius : float
                tolerance for finding node pairs, if a node pair do not respect the minimum 
                tolerance it will not considered as node pairs

            return : 
                contact_elem_dict : dict
                    dict that poitns master nodes to slaves

        '''

        if tol_radius is None:
            tol_radius = self.tol_radius

        get_node_coord = lambda node_id : np.array(self.nodes_coord[node_id])
        get_node_coord_slave = lambda node_id : np.array(self.nodes_coord_slave[node_id])
        master_nodes = self.master_nodes
        slave_nodes = self.slave_nodes
        # master points to slave # master is a key and slave is value
        contact_elem_dict = {}
        for master_node in master_nodes:
            master_coord = get_node_coord( master_node)
            min_dist = 1E8
            for slave_node in slave_nodes:
                slave_coord = get_node_coord_slave(slave_node)
                dist = np.linalg.norm(master_coord - slave_coord)
                if dist<min_dist:
                    slave_pair = slave_node
                    min_dist = dist

            if min_dist>tol_radius:
                print('It was not possible to find a slave node for master node %i. Minimum distance is %e' %(master_node,min_dist))
            else:
                contact_elem_dict[master_node] = slave_pair
                
        self.contact_elem_dict = contact_elem_dict


        self.slave_nodes = []
        for node_id in self.master_nodes:  
            try: 
                self.slave_nodes.append(self.contact_elem_dict[node_id])
            except:
                pass

        return self.contact_elem_dict
    
    def create_master_normal_dict(self, method = 'average'):
        ''' Get the normal to a node. Since there is no unique way to define the
        normal to a node, two methods are available:
        
        methods:
            first :
                compute the normal of the first element assossiated with the node
            
            average :
                compute the normal of all the elements associated with the node
                and the compute the average normal vector
        
        paramentes:
            node_id: intreturn None
                element identifier
            
            method : str
                string with 'first' or 'average'. Default value is 'average'
            
            orientation : float
                change the orientation of the normal vector either 1.0 or -1.0
        
        return
            self.master_normal_dict : dict
                dict which maps master nodes to the normal vector
        '''
        
        for master_node in self.contact_elem_dict:
            node_normal_vec = self.get_normal_at_master_node(master_node, method)
            self.master_normal_dict[master_node] = node_normal_vec
        return self.master_normal_dict
    
    def get_normal_at_master_node(self, master_node, method = 'average'):
        ''' get the normal vector of given a node
        
        parameters:
            master_node : int   
               id of the master node
            method: str
                string specifying the method to compute normal at node
        return
            normal_vector : np.array
        '''
        pass
    
    def write_files(self, filename):
        pass


class Cyclic_Contact(Contact):     
    ''' This class intend to handle cyclic contact problem,
    where master and slaves have a angule between them.
    Basically, the slave SubMesh is rotate (Virtual Slave) by the sector angule 
    and node pair are found by the minimum Euclidian distance.
    
    
    '''
    def __init__(self, master_nodes, slave_nodes, nodes_coord, connectivity=None, type='node2node', tol_radius=1e-6, sector_angle=0, 
                      unit = 'deg',dimension=3):
        
        cyclic_nodes_coord = copy.deepcopy(nodes_coord)
        if unit == 'deg':
            angle = np.deg2rad(sector_angle)
        elif unit == 'rad':
            angle = sector_angle
        else:
            raise ValueError('unit type not supported!')

        master_nodes = list(set(master_nodes))
        slave_nodes = list(set(slave_nodes))

        R = get_unit_rotation_matrix(angle,dimension)

        for slave_id in slave_nodes:
            cyclic_nodes_coord[slave_id] = R.dot(cyclic_nodes_coord[slave_id])

        self.sector_angle = sector_angle
        self.unit = unit
        master_nodes = list(set(master_nodes))
        super(Cyclic_Contact,self).__init__(master_nodes, slave_nodes, cyclic_nodes_coord , connectivity=connectivity, type=type, tol_radius=tol_radius)


class Cyclic_Constraint():
    def __init__(self,id_map_df,
                      el_df,
                      nodes_coord,
                      dirichlet_label,
                      cyclic_left_label,
                      cyclic_right_label,
                      sector_angle,
                      unit='rad',
                      tol_radius = 1.0e-3,
                      dimension=2):
    
        self.id_map_df =  id_map_df
        self.dimension = dimension
        self.dof_manager = DofManager(el_df,id_map_df)
        
        if dimension == 2:
            direction ='xy'
            print('xy direction choosen for cyclic symmetry')
        elif dimension == 3:
            direction ='xyz'
            print('xyz direction choosen for cyclic symmetry')
        else:
            raise('Dimension is not supported')
        
        self.selection_operator = create_selection_operator(id_map_df,el_df)
        
        self.theta = sector_angle
        cyclic_left_nodes = self.dof_manager.get_node_list_from_group_id(cyclic_left_label)
        cyclic_right_nodes = self.dof_manager.get_node_list_from_group_id(cyclic_right_label)
        dirichlet_nodes = self.dof_manager.get_node_list_from_group_id(dirichlet_label)

        cyclic_left_nodes = list(set(cyclic_left_nodes) - set(dirichlet_nodes))

        # creating node pairs
        contact = Cyclic_Contact(cyclic_left_nodes, cyclic_right_nodes, nodes_coord, connectivity=el_df['connectivity'], type='node2node',
                                 sector_angle=self.theta, unit=unit,tol_radius = tol_radius,dimension=dimension)


        # modifying order of nodes to have the correct node pairs for cyclic symmetry
        contact_element_dict = contact.find_node_pairs()
        cyclic_left_nodes = contact.master_nodes
        cyclic_right_nodes = contact.slave_nodes

        # get dofs
        all_dofs = self.selection_operator.list_of_all_dofs
        dir_dofs = self.selection_operator.get_union_of_dofs([dirichlet_label])

        cyclic_left = self.dof_manager.get_dofs_from_node_list(cyclic_left_nodes,direction=direction)
        cyclic_right = self.dof_manager.get_dofs_from_node_list(cyclic_right_nodes,direction=direction)


        left_dofs = OrderedSet(cyclic_left)
        right_dofs = OrderedSet(cyclic_right)

        boundary_dofs = dir_dofs | left_dofs | right_dofs
        interior_dofs = list(OrderedSet(all_dofs) - boundary_dofs)
        left_dofs = list(left_dofs)
        right_dofs = list(right_dofs)

        dof_dict = OrderedDict()
        dof_dict['d'] = dir_dofs 
        dof_dict['r'] = right_dofs
        dof_dict['l'] = left_dofs 
        dof_dict['i'] = interior_dofs

        self.dimension = dimension
        self.nc = len(left_dofs)
        self.ndofs = len(all_dofs)
        self.s = SelectionOperator(dof_dict,id_map_df)
        
        if unit == 'deg':
            self.angle = np.deg2rad(sector_angle)
        elif unit == 'rad':
            self.angle = sector_angle
        else:
            raise ValueError('unit type not supported!')
    
    def assemble_sector_operators(self):
    
        K, f = self.component.assembly_class.assemble_k_and_f()
        M = self.component.assembly_class.assemble_m()


        self.M_block = self.s.create_block_matrix(M)
        self.M_sector = self.s.assemble_matrix(M,['r','l','i'])

        K_block = self.s.create_block_matrix(K)
        self.K_sector = self.s.assemble_matrix(K,['r','l','i'])
        
        self.K =K
        self.M = M
        return self.K_sector, self.M_sector
        
    def build_complex_contraint(self,node_diam):
    
        s = self.s 
        theta = self.angle
        
        # building cyclic matrices
        #theta = -theta
        beta = node_diam*theta
        ej_beta_plus = np.exp(1J*beta)

        #building Boolean matrices
        self.Bl = Bl = s.build_B('l')
        self.Br = Br = s.build_B('r')

        T = create_voigt_rotation_matrix(self.nc, theta, dim=self.dimension)

        # Building the cyclic constraint
        C_n =  -ej_beta_plus*Br  + T.dot(Bl) 
       
        return C_n
        
    def build_complex_projection(self,node_diam):
        
        nc = self.nc
        ndofs = self.ndofs
        C_n = self.build_complex_contraint(node_diam)
        
        P_n = sparse.eye(ndofs) - 0.5*C_n.conj().T.dot(C_n)
        return P_n
        
    def build_contraint_null_space(self,node_diam):
        
        theta = self.theta
        Bl = self.Bl 
        Br = self.Br
        nc = self.nc
        ndofs = self.n_dofs
        nr = ndofs - nc 
        
        beta = node_diam*theta
        ej_beta_plus = np.exp(1J*beta)
        
        R_col1 = (ej_beta_plus.conj()*Br  + T.dot(Bl)).T
        R_col2 = sparse.vstack([0*sparse.eye(2*nc,nr-nc).tocsc(), sparse.eye(ndofs-2*nc).tocsc()]).tocsc()
        R = sparse.hstack([R_col1,R_col2]).tocsc()
        return R


class  Test_Contact(TestCase):
    def test_contact(self):

        master_nodes = [1,2] 
        slave_nodes = [4,7] 
        nodes_coord =  [[0.,0.,0.],
                        [1.,0.,0.],
                        [1.,1.,0.],
                        [0.,1.,0.],
                        [1.,0.,0.],
                        [2.,0.,0.],
                        [2.,1.,0.],
                        [1.,1.,0.]]

        cont_obj = Contact(master_nodes,slave_nodes,nodes_coord)
        contact_elem_dict = cont_obj.find_node_pairs()

        contact_elem_target = {1:4,2:7}

        self.assertDictEqual(contact_elem_target,contact_elem_dict)

    def test_cyclic_contact(self):

        master_nodes = [3,2] 
        slave_nodes = [1,0] 
        nodes_coord =  [[1.,0.,0.],
                        [2.,0.,0.],
                        [0.,2.,0.],
                        [0.,1.,0.]]
                        

        cont_obj = Cyclic_Contact(master_nodes,slave_nodes,nodes_coord,sector_angle=90)
        contact_elem_dict = cont_obj.find_node_pairs()

        slave_nodes_target = [0,1]
        contact_elem_target = {3:0,2:1}
        
        self.assertDictEqual(contact_elem_target,contact_elem_dict)
        self.assertListEqual(slave_nodes_target,cont_obj.slave_nodes)

    def test_cyclic_contact_2(self):

        master_nodes = [3,4,5] 
        slave_nodes = [1,0,2] 
        nodes_coord =  [[1.,0.,0.],
                        [1.5,0.,0.],
                        [2.,0.,0.],
                        [0.,2.,0.],
                        [0.,1.5,0.],
                        [0.,1.,0.]]
                        

        cont_obj = Cyclic_Contact(master_nodes,slave_nodes,nodes_coord,sector_angle=90)
        
        contact_elem_target = {5:0,3:2,4:1}
        slave_nodes_target = [2,1,0]
        self.assertDictEqual(contact_elem_target,contact_elem_dict)
        self.assertListEqual(slave_nodes_target,cont_obj.slave_nodes)
        
    def test_cyclic_constraint(self):

        my_dict = {}
        my_dict['idx_gmsh'] = list(range(1,17))
        my_dict['phys_group'] = [6, 7, 8, 9, 4, 4, 2, 2, 5, 5, 1, 1, 3, 3, 3, 3]
        my_dict['connectivity'] = [[0],
                                   [2],
                                   [6],
                                   [8],
                                   [0,1],
                                   [1,2],
                                   [2,5],
                                   [5,8],
                                   [8,7],
                                   [7,6],
                                   [6,3],
                                   [3,0],
                                   [0,1,4,3],
                                   [1,2,5,4],
                                   [3,4,7,6],
                                   [4,5,8,7]]

        
        el_df = pd.DataFrame(my_dict)
        id_map_df = pd.DataFrame(np.array(np.array(list(range(18)))).reshape(9,2))
        nodes_coord = np.array([[ 2.29610059e+00,  5.54327720e+00],
                                [ 3.67394040e-16,  6.00000000e+00],
                                [-2.29610059e+00,  5.54327720e+00],
                                [ 3.44415089e+00,  8.31491579e+00],
                                [ 5.51091060e-16,  9.00000000e+00],
                                [-3.44415089e+00,  8.31491579e+00],
                                [ 4.59220119e+00,  1.10865544e+01],
                                [ 7.34788079e-16,  1.20000000e+01],
                                [-4.59220119e+00,  1.10865544e+01]])

        dirichlet_label = 4
        cyclic_left_label = 2
        cyclic_right_label = 1
        sector_angle = 45
        unit='deg'
        tol_radius = 1.0e-3
        dimension=2

        cyc_obj = Cyclic_Constraint(id_map_df,
                          el_df,
                          nodes_coord,
                          dirichlet_label,
                          cyclic_left_label,
                          cyclic_right_label,
                          sector_angle,
                          unit=unit,
                          tol_radius = 1.0e-3,
                          dimension=2)

        n = 0
        Cn = cyc_obj.build_complex_contraint(n)


if __name__=='__main__':

    main()