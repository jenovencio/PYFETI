import numpy as np
import collections
import os
import matplotlib.pyplot as plt
import shutil
from pyfeti import utils
from pyfeti.src.utils import OrderedSet, Get_dofs, save_object, dict2dfmap, sysargs2keydict
from pyfeti import linalg
from pyfeti import case_generator
from pyfeti.src.feti_solver import ParallelFETIsolver
import time, logging


def create_case(number_of_div = 3, number_of_div_y=None, case_id=1):
    ''' This function create a subdomain matrices based on the number of 
    divisions.

    paramenters:
        number_of_div : int (default = 3)
            number of nodes in the x direction
        number_of_div_y : Default = None
            number of nodes in the x direction, if None value = number_of_dif
        case_id : int
            if of the case to save files

    return 
        create a directory called "matrices_{matrix shape[0]}" and store the matrices K, f, 
        B_left, B_right, B_tio, B_bottom and also the selectionOperator with the matrices indeces
    '''

    if number_of_div_y is None:
        number_of_div_y = number_of_div

    creator_obj  = utils.DomainCreator(width=number_of_div,high=number_of_div_y,x_divisions=number_of_div,y_divisions=number_of_div_y)
    creator_obj.build_elements()
    
    script_folder = os.path.join(os.path.dirname(__file__),str(case_id))
    mesh_folder = os.path.join(script_folder,'meshes')
    mesh_path = os.path.join(mesh_folder,'mesh.msh')

    try:
        creator_obj.save_gmsh_file(mesh_path)
    except:
        os.makedirs(mesh_folder, exist_ok=True)
        creator_obj.save_gmsh_file(mesh_path)

    #import mesh
    m = amfe.Mesh()
    m.import_msh(mesh_path)


    ax = amfe.plot2Dmesh(m)
    ax.set_xlim([0,number_of_div])
    ax.set_ylim([0,number_of_div_y])
    ax.set_aspect('equal')
    plt.legend('off')
    plt.savefig(os.path.join(mesh_folder,'mesh.png'))

    # creating material
    my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=1.0)

    my_system = amfe.MechanicalSystem()
    my_system.set_mesh_obj(m)
    my_system.set_domain(3,my_material)

    value = 1.0E10
    my_system.apply_neumann_boundaries(2, value, 'normal')
    id_matrix = my_system.assembly_class.id_matrix

    K, _ = my_system.assembly_class.assemble_k_and_f()
    
    _, fext = my_system.assembly_class.assemble_k_and_f_neumann()

    id_map_df = utils.dict2dfmap(id_matrix)
    try:
        connectivity = []
        for _,item in m.el_df.iloc[:, m.node_idx:].iterrows():
            connectivity.append(list(item.dropna().astype(dtype='int64')))

        m.el_df['connectivity'] = connectivity
    except:
        pass

    
    s = utils.create_selection_operator(id_map_df,m.el_df)

    neighbors_dict = {}
    neighbors_dict['right'] = 2
    neighbors_dict['left'] = 1
    neighbors_dict['top'] = 5
    neighbors_dict['bottom'] = 4
    neighbors_dict['bottom_left_corner'] = 6
    neighbors_dict['bottom_right_corner'] = 7
    neighbors_dict['top_left_corner'] = 8
    neighbors_dict['top_right_corner'] = 9

    B_dict = {}
    for key, value in neighbors_dict.items():
        B_dict[key] = s.build_B(value)

    return K, fext, B_dict, s


if __name__ == '__main__':
    help_doc = ''' 
            This python script runs a scalility test based on ParallelFETIsolver
            implemented in PyFETI

            for more information visit: [https://github.com/jenovencio/PYFETI]


            Script options
            max_mpi_size : Maximum number of mpi process in the scalability test
            divY : Number of division in the Y direction
            divX : Number of local division in the X direction

            command call
            python  create_test_case.py max_mpi_size=10 divY=10 divX=10
            '''

    import sys

    log_level = logging.INFO

    if '-h' in sys.argv:
        print(help_doc)
        exit(0)
    else:

        import amfe
        keydict = sysargs2keydict(sys.argv)
        for key, value in keydict.items():
            print('%s = %i' %(key,value))

        
        #variables
        try:
            max_mpi_size = keydict['max_mpi_size']
        except:
            max_mpi_size = 5
            print('Set max_mpi_size = %i' %max_mpi_size)

        try:
            number_of_div_y = keydict['divY']
        except:
            number_of_div_y = 5
            print('Set divY = %i' %number_of_div_y)

        try:
            local_div_x = keydict['divX']
        except:
            local_div_x = 5
            print('Set divX = %i' %local_div_x )

        
        for mpi_size in range(1,max_mpi_size+1):
            max_div_x = local_div_x*max_mpi_size
            domains_x = mpi_size
            domains_y = 1
            number_of_div_x = int(max_div_x/domains_x)

            script_folder = os.path.join(os.path.dirname(__file__),str(mpi_size),'tmp')
            logging.basicConfig(level=log_level ,filename='master.log')
            K, f, B_dict, s = create_case(number_of_div = number_of_div_x, number_of_div_y=number_of_div_y , case_id=mpi_size)
            ndof = K.shape[0]
            case_obj = case_generator.FETIcase_builder(domains_x,domains_y, K, f, B_dict, s)
            K_dict, B_dict, f_dict = case_obj.build_subdomain_matrices()
            logging.info('#################################################')
            logging.info('Starting Parallel FETI solver ..........')
            logging.info('MPI size = %i' %mpi_size)
            logging.info('Local Stiffness matrix size = (%i,%i)' %(ndof,ndof) )
            logging.info('Domains in x direction = %i' %domains_x)
            logging.info('Domains in y direction = %i' %domains_y)
            logging.info('Number of local divisions in x %i' %number_of_div_x)
            logging.info('Number of local divisions in y %i' %number_of_div_y)

            solver_obj = ParallelFETIsolver(K_dict,B_dict,f_dict,temp_folder=script_folder)
            start_time = time.time()
            sol_obj = solver_obj.solve()
            elapsed_time = time.time() - start_time
            logging.info('Parallel Solver : Elapsed time : %f.' %elapsed_time)
            os.system('rm -r ./ '+ str(mpi_size) + '/tmp/*.pkl')
            logging.info('################################################# \n\n')
