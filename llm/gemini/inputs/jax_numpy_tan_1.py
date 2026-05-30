
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tan_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array
    x = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float32 array
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: float64 precision array
    x = np.array([0.123456789, -0.987654321], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: int32 array (JAX will promote this to floating point)
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (scalar-like tensor)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Large values
    x = np.array([100.0, -1000.0, 500.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Small values near zero
    x = np.array([1e-5, -1e-6, 1e-7], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: High-dimensional array (4D)
    x = np.random.uniform(-np.pi/4, np.pi/4, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array containing multiples of pi/4 where tan values are well-known
    x = np.array([-3*np.pi/4, -np.pi/4, np.pi/4, 3*np.pi/4], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.tan_1"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tan_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tan_1'.")


check_valid('jax.numpy.tan', generated_inputs['jax.numpy.tan_1'], lib="jax", suffix=1)
