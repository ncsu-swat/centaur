
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def i0_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar), float32 positive value
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array with negative, zero, and positive values, float32
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array, float64 values
    x = np.array([[0.5, -0.5], [1.2, -1.2]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array, int32 values (will be implicitly cast)
    x = np.array([[[1, -1], [2, -2]], [[0, 3], [-3, 0]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array with small float32 values
    x = np.array([1e-5, -1e-5, 1e-4, -1e-4], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D array with purely negative float32 values
    x = np.array([[-5.0, -4.0], [-3.0, -2.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D array with random float64 values
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array with larger float32 values
    x = np.array([5.0, 8.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 0D array (scalar), int64 value
    x = np.array(-3, dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 3D array filled with zeros, float32
    x = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.i0_1"] = i0_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.i0_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.i0_1'.")


check_valid('jax.numpy.i0', generated_inputs['jax.numpy.i0_1'], lib="jax", suffix=1)
