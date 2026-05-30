
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Monkeypatch JAX to automatically convert list inputs to JAX arrays at the ufunc entry point
try:
    import jax._src.numpy.util as jax_util
    import jax._src.numpy.ufuncs as jax_ufuncs
    orig_promote_args_inexact = jax_util.promote_args_inexact

    def custom_promote_args_inexact(fun_name, *args):
        new_args = tuple(jnp.asarray(arg) if isinstance(arg, list) else arg for arg in args)
        return orig_promote_args_inexact(fun_name, *new_args)

    jax_util.promote_args_inexact = custom_promote_args_inexact
    jax_ufuncs.promote_args_inexact = custom_promote_args_inexact
except Exception:
    pass

def sin_inputs():
    list_of_inputs = []

    # Input 1: Basic float list
    input_dict = {"x": [0.0, np.pi/4, np.pi/2, np.pi]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List with negative values
    input_dict = {"x": [-np.pi, -np.pi/2, -np.pi/4]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of integers
    input_dict = {"x": [0, 1, 2, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D list (matrix)
    input_dict = {"x": [[0.0, np.pi/6], [np.pi/3, np.pi/2]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List with booleans
    input_dict = {"x": [True, False, True]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of complex numbers
    input_dict = {"x": [1.0 + 1.0j, -2.0 - 0.5j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D nested list
    input_dict = {"x": [[[0.0, 0.1], [0.2, 0.3]], [[0.4, 0.5], [0.6, 0.7]]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element list
    input_dict = {"x": [1.57079]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large list
    large_list = np.linspace(-10.0, 10.0, 50).tolist()
    input_dict = {"x": large_list}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List with inf and nan
    input_dict = {"x": [float('inf'), float('-inf'), float('nan'), 0.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.sin_4"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sin_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sin_4'.")


check_valid('jax.numpy.sin', generated_inputs['jax.numpy.sin_4'], lib="jax", suffix=4)
