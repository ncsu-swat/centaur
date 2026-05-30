
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unstack_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis 0
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 2: 2D array, axis 0
    x = np.random.randn(2, 3).astype(np.float32)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 3: 2D array, axis 1
    x = np.random.randn(2, 3).astype(np.float32)
    axis = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 4: 2D array, negative axis
    x = np.random.randn(4, 5).astype(np.float32)
    axis = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 5: 3D array, axis 0
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 6: 3D array, axis 1
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 7: 3D array, axis 2
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 8: 3D array, negative axis
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = -2
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 9: 4D array, float64, axis 3
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    axis = 3
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 10: 2D integer array, axis 0
    x = np.arange(12).reshape(3, 4).astype(np.int32)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 11: 5D array, axis -5
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axis = -5
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.unstack"] = unstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unstack'.")


check_valid('jax.numpy.unstack', generated_inputs['jax.numpy.unstack'], lib="jax", suffix=0)
