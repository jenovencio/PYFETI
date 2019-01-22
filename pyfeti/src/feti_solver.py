

class FETIsolver():
    def __init__(self,K_dict,B_dict,f_dict):
        self.K_dict = K_dict,
        self.B_dict = B_dict
        self.f_dict = f_dict
        self.x_dict = None
        self.lambda_dict = None
        self.alpha_dict = None
        
    def solve(self):
        pass
        
    def savetofiles(self):
        
        
class ParallelFETIsolver(FETIsolver):
    def __init__(self)
        super(self,FETIsolver)__init(K_dict,B_dict,f_dict)
        
    def solve(self):
        pass