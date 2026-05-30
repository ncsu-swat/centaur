
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logical_not_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    x = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D integer array with positive, negative, and zero values
    x = np.array([-1, 0, 1, 2, -5], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D integer array
    x = np.array([[0, 5], [-3, 0]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D float array
    x = np.array([[[1.5, 0.0], [0.0, -2.3]], [[0.0, 0.0], [9.1, 0.0]]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (scalar)
    x = np.array(True, dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D uint8 array
    x = np.array([0, 255, 128, 0], dtype=np.uint8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float64 array
    x = np.array([-0.0, 0.0, 1e-5, -1e-5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large multi-dimensional boolean array
    x = np.random.choice([True, False], size=(4, 4, 4))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Empty array
    x = np.array([], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D complex array
    x = np.array([0 + 0j, 1 + 0j, 0 + 1j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.logical_not"] = logical_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logical_not'.")


check_valid('jax.numpy.logical_not', generated_inputs['jax.numpy.logical_not'], lib="jax", suffix=0)
