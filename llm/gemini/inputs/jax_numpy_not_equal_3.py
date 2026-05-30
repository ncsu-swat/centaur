
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_not_equal_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D integer array and an integer
    x = np.array([1, -2, 3, 0, 5], dtype=np.int32)
    y = 3
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 2: 2D integer array and a negative integer
    x = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int64)
    y = -5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 3: 3D random integer array and an integer
    x = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    y = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 4: Float array and an integer
    x = np.array([1.0, 2.5, -3.0, 4.2], dtype=np.float32)
    y = 2
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 5: 4D integer array and an integer
    x = np.random.randint(1, 100, size=(2, 2, 2, 2)).astype(np.int32)
    y = 50
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 6: Boolean array and an integer
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 7: 0D scalar array and an integer
    x = np.array(42, dtype=np.int32)
    y = 42
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 8: Another 1D integer array and an integer
    x = np.array([10, 20, 30, 40], dtype=np.int32)
    y = 20
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 9: Large dimensions with 1s and an integer
    x = np.ones((5, 5, 5), dtype=np.int64)
    y = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 10: Float64 array with negative values and an integer
    x = np.array([[-100.0, 200.0], [-300.0, 400.0]], dtype=np.float64)
    y = -300
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.not_equal_3"] = jax_numpy_not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_3'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_3'], lib="jax", suffix=3)
