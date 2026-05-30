
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isinf_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar True)
    x = np.array(True, dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 0D array (scalar False)
    x = np.array(False, dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D boolean array
    x = np.array([True, False, True, False], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D boolean array of ones
    x = np.ones((2, 3, 2), dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D boolean array of zeros
    x = np.zeros((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Random boolean array
    x = np.random.choice([True, False], size=(5, 5)).astype(bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D boolean array of size 100
    x = np.repeat([True, False], 50).astype(bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Empty boolean array
    x = np.empty((0, 4), dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D boolean array
    x = np.ones((1, 2, 1, 2, 1), dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.isinf_4"] = isinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isinf_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isinf_4'.")


check_valid('jax.numpy.isinf', generated_inputs['jax.numpy.isinf_4'], lib="jax", suffix=4)
