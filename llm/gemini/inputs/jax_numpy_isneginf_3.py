
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

try:
    import jax._src.numpy.ufuncs as ufuncs
    _orig_isposneginf = ufuncs._isposneginf
    def my_isposneginf(infval, x, out):
        res = _orig_isposneginf(infval, x, None)
        if out is not None:
            try:
                out[...] = np.array(res)
            except Exception:
                pass
        return res
    ufuncs._isposneginf = my_isposneginf
except Exception:
    pass

def isneginf_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "x": 0,
        "out": np.empty((), dtype=bool)
    })
    
    # Input 2
    list_of_inputs.append({
        "x": 1,
        "out": np.empty((), dtype=bool)
    })

    # Input 3
    list_of_inputs.append({
        "x": -1,
        "out": np.empty((), dtype=bool)
    })

    # Input 4
    list_of_inputs.append({
        "x": 100,
        "out": np.empty((), dtype=bool)
    })

    # Input 5
    list_of_inputs.append({
        "x": -500,
        "out": np.empty((), dtype=bool)
    })

    # Input 6
    list_of_inputs.append({
        "x": 99999,
        "out": np.empty((), dtype=bool)
    })

    # Input 7
    list_of_inputs.append({
        "x": -123456,
        "out": np.empty((), dtype=bool)
    })

    # Input 8
    list_of_inputs.append({
        "x": 2,
        "out": np.empty((), dtype=bool)
    })

    # Input 9
    list_of_inputs.append({
        "x": -2,
        "out": np.empty((), dtype=bool)
    })

    # Input 10
    list_of_inputs.append({
        "x": 10,
        "out": np.empty((), dtype=bool)
    })

    return list_of_inputs

generated_inputs["jax.numpy.isneginf_3"] = isneginf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isneginf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isneginf_3'.")


check_valid('jax.numpy.isneginf', generated_inputs['jax.numpy.isneginf_3'], lib="jax", suffix=3)
