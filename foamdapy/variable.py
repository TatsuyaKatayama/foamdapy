import numpy as np

class esarray(np.ndarray):
    def __new__(cls, obj, dtype=None, state_dict:dict=None):
        self = np.asarray(obj, dtype=dtype).view(cls)
        self.state_dict= state_dict
        return self
    
    def get_st_value(self,state_name:str):
        slice_obj = self.state_dict.get(state_name)
        if slice_obj:
            return self[:,slice_obj]
        else:
            raise KeyError(f"Name '{state_name}' not found in the dictionary.")

    def set_st_value(self, state_name, value):
        slice_obj = self.state_dict.get(state_name)
        if slice_obj:
            self[:,slice_obj] = value
        else:
            raise KeyError(f"Name '{state_name}' not found in the dictionary.")