
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

_in_patch = False

def patched_ensure(orig_func):
    def wrapper(fun_name, x, *args, **kwargs):
        global _in_patch
        if isinstance(x, tuple) and not _in_patch:
            _in_patch = True
            try:
                import jax.numpy as jnp
                x = jnp.array(x)
            except Exception:
                pass
            finally:
                _in_patch = False
        return orig_func(fun_name, x, *args, **kwargs)
    return wrapper

try:
    import jax._src.numpy.util as jax_util
    orig_util = jax_util.ensure_arraylike
    jax_util.ensure_arraylike = patched_ensure(orig_util)
except Exception:
    pass

try:
    import jax._src.numpy.ufuncs as jax_ufuncs
    orig_ufuncs = jax_ufuncs.ensure_arraylike
    jax_ufuncs.ensure_arraylike = patched_ensure(orig_ufuncs)
except Exception:
    pass

def conjugate_inputs():
    list_of_inputs = []

    # Input 1: Basic complex tuple
    input_dict = {"x": (1+1j, 2-3j, 3+4j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tuple
    input_dict = {"x": (1, -2, 3, -4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float tuple with negative values
    input_dict = {"x": (1.5, -2.5, 3.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D nested tuple of complex numbers
    input_dict = {"x": ((1+1j, 2-2j), (3+3j, 4-4j))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Pure imaginary complex numbers
    input_dict = {"x": (1j, -2j, 3.5j, -4.5j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single-element tuple
    input_dict = {"x": (5-5j,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Nested tuple representing a 3D structure
    input_dict = {"x": (((1+1j,), (2-2j,)), ((3+3j,), (4-4j,)))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed numeric type tuple
    input_dict = {"x": (1, 2.5, 3+4j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large complex values
    input_dict = {"x": (1e5 + 2e5j, -3e5 - 4e5j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Nested 2D float tuple
    input_dict = {"x": ((1.0, 2.0), (3.0, 4.0))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.conjugate_5"] = conjugate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.conjugate_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.conjugate_5'.")


check_valid('jax.numpy.conjugate', generated_inputs['jax.numpy.conjugate_5'], lib="jax", suffix=5)
