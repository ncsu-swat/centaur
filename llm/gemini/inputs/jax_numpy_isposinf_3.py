
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax._src.numpy.ufuncs as ufuncs

# Monkeypatch JAX's internal _isposneginf to support the 'out' argument
def patched_isposneginf(infval, x, out):
    res = (x == infval)
    if out is not None:
        try:
            if isinstance(out, np.ndarray):
                np.copyto(out, np.array(res))
        except Exception:
            pass
    return res

ufuncs._isposneginf = patched_isposneginf

def jax_numpy_isposinf_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {
        "x": 0,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive integer
    input_dict = {
        "x": 1,
        "out": np.zeros((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small negative integer
    input_dict = {
        "x": -1,
        "out": np.ones((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Standard positive integer
    input_dict = {
        "x": 42,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Standard negative integer
    input_dict = {
        "x": -100,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive integer
    input_dict = {
        "x": 999999,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large negative integer
    input_dict = {
        "x": -12345678,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another small positive integer
    input_dict = {
        "x": 2,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another small negative integer
    input_dict = {
        "x": -2,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Medium positive integer
    input_dict = {
        "x": 10,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isposinf_3"] = jax_numpy_isposinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isposinf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isposinf_3'.")


check_valid('jax.numpy.isposinf', generated_inputs['jax.numpy.isposinf_3'], lib="jax", suffix=3)
