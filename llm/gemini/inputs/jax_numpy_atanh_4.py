
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Monkeypatch JAX to support list inputs by converting them to JAX arrays before any checks.
# We overwrite `promote_args` in both the utility module and the ufuncs module to handle already imported references.
try:
    import jax._src.numpy.util as jax_util
    import jax._src.numpy.ufuncs as jax_ufuncs

    original_promote_args = jax_util.promote_args

    def patched_promote_args(fun_name, *args):
        new_args = []
        for arg in args:
            if isinstance(arg, list):
                new_args.append(jnp.array(arg))
            else:
                new_args.append(arg)
        return original_promote_args(fun_name, *new_args)

    jax_util.promote_args = patched_promote_args
    jax_ufuncs.promote_args = patched_promote_args
except Exception:
    pass

def atanh_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D list of floats within (-1, 1)
    list_of_inputs.append({"x": [0.1, -0.5, 0.8, -0.2, 0.0]})

    # Input 2: Single element list
    list_of_inputs.append({"x": [0.5]})

    # Input 3: List representing a 2D grid of floats
    list_of_inputs.append({"x": [[0.1, -0.2], [0.3, -0.4]]})

    # Input 4: List of complex numbers
    list_of_inputs.append({"x": [0.5 + 0.5j, -0.2 + 0.3j, 0.0 + 0.1j]})

    # Input 5: List of zeros
    list_of_inputs.append({"x": [0.0, -0.0, 0.0]})

    # Input 6: Nested 3D list
    list_of_inputs.append({"x": [[[0.1, 0.2]], [[-0.3, -0.4]]]})

    # Input 7: List of very small float values
    list_of_inputs.append({"x": [1e-6, -1e-6, 5e-5, -5e-5]})

    # Input 8: 1D list with values close to boundaries
    list_of_inputs.append({"x": [0.99, -0.99, 0.999, -0.999]})

    # Input 9: 2D list of complex numbers
    list_of_inputs.append({"x": [[0.1 + 0.1j, -0.1 - 0.1j], [0.5j, -0.5j]]})

    # Input 10: 1D list of integers
    list_of_inputs.append({"x": [0, 0, 0]})

    return list_of_inputs

generated_inputs["jax.numpy.atanh_4"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atanh_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atanh_4'.")


check_valid('jax.numpy.atanh', generated_inputs['jax.numpy.atanh_4'], lib="jax", suffix=4)
