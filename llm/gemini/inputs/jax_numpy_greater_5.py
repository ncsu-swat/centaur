
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def greater_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float array, positive integer
    x = 5
    y = np.array([1.0, 6.0, 5.0, 4.5, 10.2], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2: 2D integer array, zero integer
    x = 0
    y = np.array([[-1, 2, 0], [3, -4, 5]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3: 3D random float array, negative integer
    x = -10
    y = np.random.uniform(-20, 0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4: Scalar tensor, positive integer
    x = 42
    y = np.array(42, dtype=np.int64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5: 1D integer array with large values, large integer
    x = 1000
    y = np.array([500, 1500, 1000, 2000, 100], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6: 4D random integer array, positive integer
    x = 3
    y = np.random.randint(-5, 10, size=(2, 2, 2, 2)).astype(np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7: 1D float16 array, negative integer
    x = -1
    y = np.array([-2.5, -1.0, 0.5, 1.5], dtype=np.float16)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8: 2D float64 array, positive integer
    x = 15
    y = np.random.uniform(10, 20, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9: 3D array with broadcast shape, negative integer
    x = -5
    y = np.random.randint(-10, 10, size=(4, 1, 5)).astype(np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10: 1D array of zeros and ones, zero integer
    x = 0
    y = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.greater_5"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_5'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_5'], lib="jax", suffix=5)
