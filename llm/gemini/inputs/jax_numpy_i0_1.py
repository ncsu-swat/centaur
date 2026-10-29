
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def i0_inputs():
    list_of_inputs = []

    # Input 1: 1D array of floats with positive, negative, and zero values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 0D scalar array (float64)
    x = np.array(3.5, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array of integers
    x = np.array([[-3, -2, -1], [0, 1, 2]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array of floats
    x = np.random.uniform(-5.0, 5.0, size=(4, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array of float64
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array of float16
    x = np.random.uniform(-2.0, 2.0, size=(2, 2, 3, 3)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array of int64
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Array with large values
    x = np.array([10.0, 20.0, 50.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array containing zeros
    x = np.zeros((3, 5), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D array with fractional/small values
    x = np.array([-0.01, -0.001, 0.0, 0.001, 0.01], dtype=np.float64)
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
