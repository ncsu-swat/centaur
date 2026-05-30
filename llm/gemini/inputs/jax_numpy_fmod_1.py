
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmod_inputs():
    list_of_inputs = []

    # Input 1: Small 2D arrays with positive and negative floats
    x1 = np.array([[3.0, -1.0, 4.0], [8.0, 5.0, -2.0]], dtype=np.float32)
    x2 = np.array([[2.0, 3.0, -5.0], [2.0, 3.0, -5.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 1D arrays, int32 with positive and negative integers
    x1 = np.array([10, -10, 10, -10, 5], dtype=np.int32)
    x2 = np.array([3, 3, -3, -3, 2], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: Broadcasting 2D and 1D arrays, float32
    x1 = np.array([[3.5, -1.2, 4.7], [8.1, 5.5, -2.9]], dtype=np.float32)
    x2 = np.array([2.0, 3.0, -5.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 3D arrays, float64
    x1 = np.random.uniform(-10.0, 10.0, (2, 2, 2)).astype(np.float64)
    x2 = np.random.uniform(1.0, 5.0, (2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: 1-element 1D arrays, float32
    x1 = np.array([7.5], dtype=np.float32)
    x2 = np.array([2.5], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Large 2D arrays, float32
    x1 = np.random.uniform(-100.0, 100.0, (10, 10)).astype(np.float32)
    x2 = np.random.uniform(1.0, 10.0, (10, 10)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: int64 arrays, 2D
    x1 = np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]], dtype=np.int64)
    x2 = np.array([[7, 8, 9], [7, 8, 9], [7, 8, 9]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: Broadcasting (1, 5) and (5, 1) arrays, float32
    x1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    x2 = np.array([[1.5], [2.5], [3.5], [4.5], [5.5]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Small fractional values, float32
    x1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    x2 = np.array([0.03, 0.07, 0.11], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: 4D arrays with broadcasting, float32
    x1 = np.random.uniform(-5.0, 5.0, (2, 1, 3, 2)).astype(np.float32)
    x2 = np.random.uniform(1.0, 2.0, (1, 2, 1, 2)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.fmod_1"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_1'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_1'], lib="jax", suffix=1)
