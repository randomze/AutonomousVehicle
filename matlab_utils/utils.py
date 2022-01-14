import matlab.engine
import os
from scipy.io import savemat

cur_engs = matlab.engine.find_matlab()
if len(cur_engs) == 0:
    eng = matlab.engine.start_matlab()
else:
    eng = matlab.engine.connect_matlab(name=cur_engs[0])
del cur_engs

def get_engine() -> matlab.engine.MatlabEngine:
    """Returns the matlab engine corresponding to the current matlab session.
    """
    return eng

def to_workspace(*dicts_w_vars) -> None:
    """Receives dictionaries with variables and their values and saves them to 
    the matlab workspace.

    Args:
        *dicts_w_vars: dictionaries with variables and their values.
    """
    for d in dicts_w_vars:
        for key, value in d.items():
            eng.workspace[key] = value

def big_var_to_workspace(var_name: str, var_value) -> None:
    """Saves a variable to the matlab workspace though saving it to a file and
    loading it using the matlab engine. This is surprisingly faster than saving
    it directly to the workspace, for big variables (such as images).
    
        Args:
            var_name: name of the variable.
            var_value: value of the variable.
    """
    path = os.path.abspath(f'{var_name}.mat')
    savemat(path, {var_name: var_value})
    eng.eval(f"load('{path}');", nargout=0) 
    os.remove(f'{var_name}.mat')
    