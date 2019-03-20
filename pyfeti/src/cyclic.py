import numpy as np

def maptoglobal(modes_dict,s):
    global_dofs = s.P.shape[0]
    new_modes_dict = {}
    for i in modes_dict:
        modes = modes_dict[i]
        local_dofs, num_modes = modes.shape
        zeros = np.zeros((global_dofs - local_dofs,num_modes))
        local_modes = np.vstack((zeros,modes))
        global_modes = s.P.T.dot(local_modes)
        new_modes_dict[i] = global_modes
    return new_modes_dict