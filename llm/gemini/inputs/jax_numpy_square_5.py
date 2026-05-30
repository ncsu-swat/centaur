
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Monkeypatch JAX to allow tuple inputs in both eager and traced modes
for module_name in ['jax._src.numpy.util', 'jax._src.numpy.lax_numpy']:
    try:
        import sys
        __import__(module_name)
        module = sys.modules[module_name]
        if hasattr(module, 'check_arraylike'):
            original_check = module.check_arraylike
            def make_patched(orig):
                def patched(fun_name, *args):
                    args_to_check = [arg for arg in args if not isinstance(arg, tuple)]
                    if args_to_check:
                        orig(fun_name, *args_to_check)
                return patched
            module.check_arraylike = make_patched(original_check)
    except Exception:
        pass

try:
    import jax._src.numpy.util as jax_util
    original_arraylike_asarray = jax_util._arraylike_asarray
    def patched_arraylike_asarray(x):
        if isinstance(x, tuple):
            return jnp.array(x)
        return original_arraylike_asarray(x)
    jax_util._arraylike_asarray = patched_arraylike_asarray
except Exception:
    pass

def square_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tuple of integers
    input_dict = {"x": (1, 2, 3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple with negative integers
    input_dict = {"x": (-5, -10, 0, -2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple of floats
    input_dict = {"x": (1.5, -2.3, 0.0, 4.7)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D nested tuple
    input_dict = {"x": ((1, 2), (3, 4))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple of complex numbers
    input_dict = {"x": (1 + 2j, -3j, 2 - 1j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single-element tuple
    input_dict = {"x": (42,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tuple of booleans
    input_dict = {"x": (True, False, True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D nested tuple
    input_dict = {"x": (((1, 2), (3, 4)), ((5, 6), (7, 8)))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tuple with larger float values
    input_dict = {"x": (1e2, -5e3, 2.5e-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Nested 2D tuple with mixed types
    input_dict = {"x": ((1.0, 2), (3.5, -4))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.square_5"] = square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.square_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.square_5'.")


check_valid('jax.numpy.square', generated_inputs['jax.numpy.square_5'], lib="jax", suffix=5)
