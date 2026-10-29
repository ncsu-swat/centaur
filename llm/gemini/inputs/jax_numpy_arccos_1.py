
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arccos_inputs():
    list_of_inputs = []

    # Input 1: 1D array with values in standard range [-1, 1], float32
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array, float64
    x = np.array([[-0.8, -0.2], [0.3, 0.9]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 0D scalar array, float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array with uniform random values in [-1, 1], float32
    x = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D integer array (will be promoted to inexact dtype)
    x = np.array([-1, 0, 1], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D complex array, complex64
    x = np.array([4.0 - 1.0j, -2.0 + 3.0j, 0.5 + 0.5j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D complex array, complex128
    x = np.array(
        [[1.0 + 2.0j, -0.5 - 0.5j], [3.0 + 0.0j, 0.0 - 1.5j]],
        dtype=np.complex128,
    )
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Array containing boundary values and out-of-domain real values (produces nan)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: float16 array
    x = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 4D array with values in [-1, 1], float32
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Empty array, float32
    x = np.array([], dtype=np.float32).reshape(0, 5)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.numpy.arccos_1"] = arccos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccos_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccos_1'.")


check_valid('jax.numpy.arccos', generated_inputs['jax.numpy.arccos_1'], lib="jax", suffix=1)
