import numpy as np
import collections
import os
import matplotlib.pyplot as plt
import shutil
from pyfeti import utils
from pyfeti.src.utils import OrderedSet, Get_dofs, save_object, dict2dfmap, sysargs2keydict
from pyfeti import linalg
from pyfeti import case_generator
from pyfeti.src import feti_solver
import time, logging, json


def create_case(width = 100, heigh=100, divX=100, divY=100, case_id=1,save_fig=False):
    ''' This function create a subdomain matrices based on the number of 
    divisions.

    paramenters:
        width : float, Default = 100 [mm]
            width of the 2D body
        heigh : float, Default = 100 [mm]
            heigh of the 2D body
        divY : int, Default = 100 
            divisions in Y direction
        divX : int, Default = 100 
            divisions in X direction
        case_id : int
            if of the case to save files

    return 
        create a directory called "matrices_{matrix shape[0]}" and store the matrices K, f, 
        B_left, B_right, B_tio, B_bottom and also the selectionOperator with the matrices indeces
    '''

    

    creator_obj  = utils.DomainCreator(width=width*1.0e-3,heigh=heigh*1.0e-3,x_divisions=int(divX)+1,y_divisions=int(divY)+1)
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

    if save_fig:
        ax = amfe.plot2Dmesh(m)
        ax.set_xlim([0,width*1.0e-3])
        ax.set_ylim([0,heigh*1.0e-3])
        ax.set_aspect('equal')
        ax.set_xlabel('Width [m]')
        ax.set_ylabel('Heigh [m]')
        plt.legend('off')
        
        plt.savefig(os.path.join(mesh_folder,'mesh.png'))


    # creating material
    my_material = amfe.KirchhoffMaterial(E=210.0E5, nu=0.3, rho=7.86E-9, plane_stress=True, thickness=1.0e-3)

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

def create_salamon_job(directory=None, job_name='submit.sh',nnodes=1,queue='qexp',hours=1,minutes=00,**kwargs):
    
    if directory is None:
        directory = os.getcwd()

    commands = \
    '''
    # variables
    scriptfile=test.sh
    nnodes=%i
    ncpus=24
    queue=%s
    hours=%i
    min=%i

    echo -------------------------------------------------
    echo -----------  run PyFETI scalability -------------
    echo -------------------------------------------------
    echo author: Guilherme Jenovencio

    echo creating a local script for scalability ...
    echo script name = $scriptfile
    echo cd $PWD>$scriptfile
    echo ml Anaconda3>>$scriptfile
    echo ml MPICH>>$scriptfile
    echo bash run_mpi.sh>>$scriptfile
    echo queue  selected  = $queue
    echo nodes = $nnodes
    echo CPU = $ncpus
    echo Hours = $hours
    echo minutes = $min
    qsub -A  $project_id -q $queue -l select=$nnodes:ncpus=$ncpus,walltime=$hours:$min:00 $scriptfile
    ''' %(nnodes,queue,hours,minutes)

    job_path = os.path.join(directory, job_name)
    with open(job_path,'w') as f:
        f.write(commands)

    return job_path

def factorize_mpi(mpi_size):
    ''' Factorize mpi in the multiplication of
    the biggest interger numbers

    e.g   9 -> 3*3
          6 -> 3*2
          4 -> 2*2
          5 -> 2*2 and change mpi size
    Parameters:
        mpi_size : int

    reuturn 
        factor1 : int
        factor2 : int
        mpi_size : int
            as factor1*factor2
    ''' 
    factor1 = int(np.sqrt(mpi_size))
    factor2 = int(mpi_size/factor1)
    if not factor1>=factor2:
        factor1,factor2 = factor2, factor1

    if int(factor1*factor2)!=mpi_size:
        logging.warning('Changing mpi size to fit the best rectangular subdomains')
        mpi_size = int(factor1*factor1)
    return factor1,factor2,mpi_size

if __name__ == '__main__':
    help_doc = ''' 
            This python script runs a scalabity test based on ParallelFETIsolver
            implemented in PyFETI

            for more information visit: [https://github.com/jenovencio/PYFETI]

                            W = width
               __ __ __ __ __ __ __ __ __ __ __ __
              |                                   |
              |                                   |
            H |                                   |
              |                                   |
              |__ __ __ __ __ __ __ __ __ __ __ __|

            Script options
            W  : float value for the width in [mm] of the 2D plane-stress body. Default = 60
            H  : float value for the heigh in [mm] of the 2D plane-stress body. Default = 60
            divY : Number of division in the Y direction, Default = 24
            divX : Number of local division in the X direction, Default = 24
            domainX  : list of domains in the X direciton. Default = [1,2,3]
            domainY  : list of domains in the Y direciton. Default = [1,2,3]
            method : Method to compute the local pseudoinverse, Default = splusps (splusps also avaliable)
            FETI_algorithm : Type of FETI algorithm SerialFETIsolver of ParallelFETIsolver,  Default = ParallelFETIsolver
            tol : tolerance of PCPG error norm, Default = 1.0E-5
            precond : Preconditioner type : Default - Identity (options: Lumped, Dirichlet, LumpedDirichlet, SuperLumped)
            square : create a square of retangular domains depended on the mpi, Default : False
            BC_type : type of Neumman B.C, Defult = RX, options {RX,G} RX is force in x at the right domains, G is gravity in Y
            strong : Boolean variable, if True perform strong scalability, if False, perform weak scalability, Default = True
            loglevel : INFO, DEBUG, ERROR, WARNING, CRITICAL. Default = INFO
            launcher_only : Boolean variable to create scripts to without launch mpi : Default = False
            delete_files : Boolean variable to delete *.pkl files after mpirun : Default = True
            salomon : {} dict with salomon paramenters e.g. {'queue':'qexp','ncpus' : 24, 'default_time':30, 'effectivity': 0.7}.  Default = {}
                         'default_time' is given in minutes, an estimation of required HPC time will be computed based on it.
            example of command call:
            > python  create_test_case.py W=60 H=60 domainX=[2,3] domainX=[1,1] 

            '''

    default_dict = {'loglevel' : 'INFO',
                    'strong'  : True,
                    'FETI_algorithm' : 'ParallelFETIsolver',
                    'square' : True,
                    'BC_type' : 'RX',
                    'precond' : None,
                    'tol' : 1.0E-5,
                    'method' : 'splusps',
                    'domainX' : [1,2,3],
                    'domainY' : [1,2,3],
                    'divY' : 24,
                    'divX' : 24,
                    'launcher_only' : False,
                    'delete_files' : True,
                    'W' : 60.0,
                    'H' : 60.0,
                    'salomon' : {}}

    salamon_defaut = {'queue':'qexp','default_time':30,'ncpus':24, 'effectivity': 0.5}

    header ='#'*50
    import sys
    from datetime import datetime
    curdir = os.getcwd()
    
    if '-h' in sys.argv:
        print(help_doc)
        exit(0)
    else:

        import amfe

        # transform system arguments in python dict
        keydict = sysargs2keydict(sys.argv)
        
        # update variables with system argument dict
        default_dict.update(keydict)

        # add default dict to local variables
        locals().update(default_dict)

        if len(domainY)!=len(domainX):
            logging.warning('DomainY list with different length of DomainX. Setting new DomainY')
            domainY = [domainY[0]]*len(domainX)
            logging.warning(('new DomainY = ',domainY))
        
        
        max_mpi_size = (max(domainX)*max(domainY))
        min_mpi_size = (min(domainX)*min(domainY))

        date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        scalability_folder = os.path.join(curdir,date_str)
        os.mkdir(scalability_folder)

        # change to scalability local folder
        os.chdir(scalability_folder)

        LOG_FORMAT = "%(levelname)s : %(message)s"
        log_level = getattr(logging,loglevel)
        logging.basicConfig(level=log_level ,filename='master_' + date_str  + '.log', format=LOG_FORMAT)
        logging.info(header)
        logging.info('#####################    SCALABILITY TEST  #######################')
        logging.info(header)
        logging.info(datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
        
        if strong:
            logging.info('Perform STRONG parallel scalability.')
        else:
            logging.info('Perform WEAK parallel scalability.')
        logging.info('Square  = %s' %str(square ))
        logging.info('Neumann B.C type  = %s' %BC_type)
        if precond is None:
            logging.info('Preconditioner type  = %s' %'Identity')
        else:
            logging.info('Preconditioner type  = %s' %precond)
        logging.info('PCPG tolerance  = %2.2e' %tol)
        logging.info('Set pseudoinverse method  = %s' %method)
        logging.info('Set divX = %i' %divX)
        logging.info('Set divY = %i' %divY)
        logging.info('Set max_mpi_size = %i' %max_mpi_size)
        logging.info('Set min_mpi_size = %i' %min_mpi_size)
        


        for domain_x,domain_y in zip(domainX,domainY):
            
            mpi_size = domain_x*domain_y
            # subdomain dimensions
            w = W/domain_x
            h = H/domain_y

            if strong:
                div_x = divX/domain_x
                div_y = divY/domain_y
                
            else:
                div_x = divX
                div_y = divY
                
            domain_size = div_x*div_y*2 # number of domains x num of nodes x dof per node    
            logging.info('Local Domain : size (%i,%i)' %(domain_size,domain_size))
            
            logging.info('Local Domain : number of local divisions in (X,Y) = (%i,%i)' %(div_x,div_y))
            logging.info(header)
            logging.info('########################     MPI size  : %i    #####################' %mpi_size)
            logging.info(header)
            logging.info('Date - Time = ' + datetime.now().strftime('%Y-%m-%d - %H:%M:%S'))
            script_folder = os.path.join(os.path.dirname(__file__),str(mpi_size),'tmp')

            logging.info('Domains in x direction = %i' %domain_x)
            logging.info('Domains in y direction = %i' %domain_y)
            logging.info('Number of local divisions in x %i' %div_x)
            logging.info('Number of local divisions in y %i' %div_y)
            
            logging.info(header)
            logging.info('# AMFE log : Assembling local matrices')     
            logging.info('Local Domain width [mm] = %2.2f' %w)
            logging.info('Local Domain heigh [mm] = %2.2f' %h)       
            K, f, B_dict, s = create_case(width = w, heigh=h, divX=div_x, divY=div_y , case_id=mpi_size, save_fig=True)
            ndof = K.shape[0]
            case_obj = case_generator.FETIcase_builder(domain_x,domain_y, K, f, B_dict, s, BC_type=BC_type)
            K_dict, B_dict, f_dict = case_obj.build_subdomain_matrices()
            logging.info('# END AMFE log')    
            logging.info(header)

            logging.info(header)
            logging.info('# Starting Parallel FETI solver ..........')
            logging.info(header)
            logging.info('{"MPI_size" : %i}' %mpi_size)
            logging.info('{"Local_Stiffness_matrix_size" = (%i,%i)}' %(ndof,ndof) )
            

            # solver parameters
            pseudoinverse_kargs={'method':method,'tolerance': 1.0E-8}
            dual_interface_algorithm = 'PCPG'

            logging.info('{"dual_interface_algorithm" :  "%s"}' %dual_interface_algorithm)
            logging.info('{"pseudoinverse_method" : "%s"}' %pseudoinverse_kargs['method'])
            logging.info('{"pseudoinverse_tolerance" : %2.2e}' %pseudoinverse_kargs['tolerance'])
            logging.info('{"Dual interface tolerance" : %2.2e}' %tol)

            try:
                FETIsolver = getattr(feti_solver, FETI_algorithm)
                solver_obj = FETIsolver(K_dict,B_dict,f_dict,temp_folder=script_folder,
                                                pseudoinverse_kargs=pseudoinverse_kargs,
                                                dual_interface_algorithm=dual_interface_algorithm,tolerance=tol,
                                                precond_type=precond,launcher_only=launcher_only)

                
                start_time = time.time()
                solution_obj = solver_obj.solve()
                

                if not launcher_only:

                    elapsed_time = time.time() - start_time
                    logging.info('{"Parallel Solver" : %f} #Elapsed time (s)' %elapsed_time)
                    solution_obj.local_matrix_time
                    solution_obj.time_PCPG 
                    logging.info('{"Interface_size" : %i}' %len(solution_obj.interface_lambda))
                    logging.info('{"Primal_variable_size" : %i}' %len(solution_obj.displacement))
                    logging.info('{"Course_problem_size" : %i}' %len(solution_obj.alpha))
                    logging.info('{"PCPG_iterations" : %i}' %solution_obj.PCGP_iterations)
                    logging.info('{"PCPG_residual" : %6.4e}' %solution_obj.projected_residual)
                    logging.info('{"Global_FETI_solver" : %f} #Elapsed time (s)' %solution_obj.solver_time)
                    logging.info('{"Local_matrix_preprocessing" : %f} #Elapsed time (s)' %solution_obj.local_matrix_time)
                    logging.info('{"PCPG" : %f} #Elapsed time (s)' %solution_obj.time_PCPG)
                    logging.info('Date - Time = ' + datetime.now().strftime('%Y-%m-%d - %H:%M:%S'))
                    logging.info(header)
                    logging.info('END OF MPI size : %i' %mpi_size)
                    logging.info(header)
                    logging.info('\n\n\n')
                    if delete_files:
                        os.system('rm -r ./ '+ str(mpi_size) + '/tmp/*.pkl')
                else:
                    elapsed_time = time.time() - start_time
                    logging.info('{"Preprocessing_time" : %f} #Elapsed time (s)' %elapsed_time)
                    
                    simulation_folder = os.path.join(scalability_folder,solver_obj.manager.temp_folder)
                    local_dict = {}
                    local_dict['simulation_folder'] = simulation_folder
                    local_dict['preprocessing_time[s]'] = elapsed_time
                    local_dict['preprocessing_time[s]'] = elapsed_time
                    local_dict['case_info'] = {'div_x':div_x,'div_y':div_y,'domain_x':domain_x, 
                                               'domain_y':domain_y, 'Kshape' : K.shape,
                                                'W' : W, 'H' : H, 'width' : w, 'heigh' : h }
                    
                    if salomon:

                        salamon_defaut.update(salomon)
                        ncpus = salamon_defaut['ncpus']
                        default_time = salamon_defaut['default_time']
                        effectivity = salamon_defaut['effectivity']

                        # Heuristic estimation for Jobs
                        time_reduction_factor = (min_mpi_size/mpi_size)/(effectivity)
                        time_in_minute = np.ceil(default_time*time_reduction_factor)
                        nnodes = int(mpi_size//ncpus) + int(bool(mpi_size%ncpus))

                        salomon['nnodes'] = nnodes
                        salomon['hours'] = int(time_in_minute//60)
                        salomon['minutes'] =  int(time_in_minute%60)
                        
                        salamon_defaut.update(salomon)
                        job_path = create_salamon_job(**salamon_defaut)
                        local_dict['job_path'] = job_path


                    case_dict = {}
                    case_dict['case'] = {mpi_size : local_dict}

                    with open(os.path.join(curdir, date_str + '.json'),'a') as f:
                        json.dump(case_dict,f)
                        f.write('\n')
                        
            except:
                logging.error('Parallel solver Error!')

            # back to original folder
            os.chdir(scalability_folder)
