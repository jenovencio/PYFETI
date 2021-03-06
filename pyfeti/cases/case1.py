import sys
import os

sys.path.append('../..')
cases_folder = os.path.dirname(os.path.realpath(__file__))
casefiles_folder = os.path.join(cases_folder, 'case1files')
sys.path.append(cases_folder)
sys.path.append(casefiles_folder)

from pyfeti.src.utils import load_object


K1 = load_object(os.path.join(casefiles_folder,'K1.pkl'))
K2 = load_object(os.path.join(casefiles_folder,'K2.pkl'))
B1_dict = load_object(os.path.join(casefiles_folder,'B1_dict.pkl'))
B2_dict = load_object(os.path.join(casefiles_folder,'B2_dict.pkl'))
global_to_local_dict_1 = load_object(os.path.join(casefiles_folder,'global2local_1.pkl'))
global_to_local_dict_2 = load_object(os.path.join(casefiles_folder,'global2local_2.pkl'))
dofs_dict = load_object(os.path.join(casefiles_folder,'dofs_dict.pkl'))
map_dofs = load_object(os.path.join(casefiles_folder,'map_dofs.pkl'))
K_global = load_object(os.path.join(casefiles_folder,'K_global.pkl'))[0]