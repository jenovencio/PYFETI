import matplotlib.pyplot as plt
from pyfeti.src.feti_solver import Solution 
from pyfeti.src.utils import load_object
import json, os
import pandas as pd


def read_json(json_file):
    ''' read json 
    '''
    with open(json_file,'r') as f:
        case_dict = {}
        for lines in f.readlines():
             case_dict.update(json.loads(lines)['case'])
    return case_dict

def read_solution(solution_path):
    ''' read solution file
    '''
    solution_obj = load_object(solution_path,tries=1,)

    return solution_obj

def read_case_dict(scalability_folder, case_dict):
    solution_dict = {}
    for key, item in case_dict.items():
        solution_path = os.path.join(scalability_folder,item['simulation_folder'],'solution.pkl')
        obj = read_solution(solution_path)
        if obj is not None:
            solution_dict[key] = obj
       
    return solution_dict


def solution_dict_to_df(solution_dict):

    domains = list(solution_dict.keys())
    col = []
    col.extend(domains)
    df = pd.DataFrame(columns=col)

    mapdict = {}
    mapdict['Total Time [s]'] = 'solver_time'
    mapdict['Number of Iterations'] = 'PCGP_iterations'
    mapdict['PCPG time [s]'] = 'time_PCPG'
    mapdict['Preprocessing [s]'] = 'local_matrix_time'
    mapdict['Interface size'] = 'lambda_size'
    mapdict['Kernel size'] = 'alpha_size'
    for row_label, atribute in mapdict.items():
        row = []
        for key in domains:
            row.extend(['{:2.2f}'.format(getattr(solution_dict[key],atribute))])
        df.loc[row_label] = row
    return df
    
    
def solution_dict_2_pie(solution_dict):
    pie_dict_ = {}
    iter_dict = {}
    for key, item in solution_dict.items():
        pie_dict_[key] = {}
        iter_dict[key] = {}
        pie_dict = pie_dict_[key]
        n = item.PCGP_iterations
        v1, v2, v3, v4, v5, v6 = [0.0]*6
        for i in range(n): 
            v1 += item.info_dict[i]['elaspsed_time_projection']
            v2 += item.info_dict[i]['elaspsed_time_precond']
            v3 += item.info_dict[i]['elaspsed_time_beta']
            v4 += item.info_dict[i]['elaspsed_time_F_action']
            v5 += item.info_dict[i]['elaspsed_time_alpha']
            v6 += item.info_dict[i]['elaspsed_time_iteration']

        pie_dict['avg_proj'] = v1/v6
        pie_dict['avg_precond'] = v2/v6
        pie_dict['avg_beta'] = v3/v6
        pie_dict['avg_F'] = v4/v6
        pie_dict['avg_alpha'] = v5/v6
        pie_dict['overhead'] = (1.0 - (v1+v2+v3+v4+v5)/v6)
        iter_dict[key] = v6/n

    pie_dict = pie_dict_
    return pie_dict, iter_dict

if __name__ == '__main__':
    
    scalability_folder = '/home/ge72tih/dev/scalability'
    json_file = '2019_05_28_13_08_17.json'

    json_path = os.path.join(scalability_folder,json_file)


    # reading json file
    case_dict = read_json(json_path)

    case_folder = []
    for key, item  in case_dict.items():
        case_folder.append(item['simulation_folder'])
        
    solution_dict = {}
    for key, item in case_dict.items():
        solution_path = os.path.join(scalability_folder,item['simulation_folder'],'solution.pkl')
        solution_dict[key] = read_solution(solution_path)

    plt.plot(solution_obj.proj_r_hist,'-o')
    plt.xlabel('iterations')
    plt.ylabel('Projected Residual')
    plt.title('Dual interface size = %i' %solution_obj.lambda_size)

    plt.show()

    x=1