import matplotlib.pyplot as plt
from pyfeti.src.feti_solver import Solution 
from pyfeti.src.utils import load_object


def read_json(json_file):
    ''' read json 
    '''
    with open(json_file,'r') as f:
        case_dict = {}
        for lines in f.readlines():
             case_dict.update(eval(lines)['case'])
    return case_dict

def read_solution(solution_path):
    ''' read solution file
    '''
    solution_obj = load_object(solution_path)

    return solution_obj

if __name__ == '__main__':
    json_file = '/home/ge72tih/dev/scalability/tmp/2019_05_24_16_01_40.json'
    case_dict = read_json(json_file)

    solution_file = '/home/ge72tih/dev/scalability/tmp/2019_05_24_16_01_40/48/tmp/solution.pkl'
    solution_file = '/home/ge72tih/dev/PYFETI/scripts/scalability/2019_05_28_03_36_15/9/tmp/solution.pkl'

    solution_obj = read_solution(solution_file)

    plt.plot(solution_obj.proj_r_hist,'-o')
    plt.xlabel('iterations')
    plt.ylabel('Projected Residual')
    plt.title('Dual interface size = %i' %solution_obj.lambda_size)

    plt.show()

    x=1