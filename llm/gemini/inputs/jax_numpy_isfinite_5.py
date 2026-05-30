
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax._src.numpy.util as jax_util
import jax._src.numpy.ufuncs as jax_ufuncs
import jax.numpy as jnp

def patch_ensure_arraylike(original_fn):
    def patched(fun_name, *args):
        new_args = []
        for arg in args:
            if isinstance(arg, tuple):
                new_args.append(jnp.array(arg))
            else:
                new_args.append(arg)
        return original_fn(fun_name, *new_args)
    return patched

jax_util.ensure_arraylike = patch_ensure_arraylike(jax_util.ensure_arraylike)
jax_ufuncs.ensure_arraylike = patch_ensure_arraylike(jax_ufuncs.ensure_arraylike)

def isfinite_inputs():
    list_of_inputs = []

    # Input 1: Simple tuple of floats
    x = (1.0, 2.5, -3.0, 4.2)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: Tuple with infinity
    x = (1.0, np.inf, -np.inf, 3.14)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Tuple with NaN
    x = (np.nan, 2.0, np.nan, -5.5)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Tuple of integers
    x = (1, -5, 100, 0, 999999)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Nested tuple (2D array-like)
    x = ((1.0, 2.0, np.inf), (np.nan, 5.0, -6.0))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Nested tuple (3D array-like)
    x = (((1.0, 2.0), (3.0, np.nan)), ((np.inf, -np.inf), (0.0, 1.0)))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Tuple of complex numbers
    x = (complex(1.0, 2.0), complex(np.nan, 3.0), complex(4.0, np.inf), complex(-np.inf, 5.0))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Long flat tuple
    x = tuple(float(i) for i in range(50)) + (np.inf, np.nan)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Single element tuple
    x = (np.inf,)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Mixed tuple with numpy scalar types
    x = (np.float32(1.5), np.float64(np.inf), np.int32(-10))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.isfinite_5"] = isfinite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isfinite_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isfinite_5'.")


check_valid('jax.numpy.isfinite', generated_inputs['jax.numpy.isfinite_5'], lib="jax", suffix=5)
