import matplotlib.pyplot as plt
from pyfeti.src.feti_solver import Solution 
from pyfeti.src.utils import load_object
import json, os


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
        solution_dict[key] = read_solution(solution_path)
    return solution_dict

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