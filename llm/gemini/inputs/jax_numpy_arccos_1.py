
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arccos_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar) within [-1, 1], float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array with values in [-1, 1], float32
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array with values in [-1, 1], float64
    x = np.linspace(-1.0, 1.0, 9).reshape(3, 3).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array with some values outside [-1, 1] to test nan behavior, float32
    x = np.array([[[ -2.0, 0.0], [0.5, 2.0]], [[-0.1, 0.1], [1.5, -1.5]]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex array, complex64
    x = np.array([1.0 + 1.0j, -2.0 - 3.0j, 0.5j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D integer array (will be cast to float)
    x = np.array([[-1, 0], [1, 0]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D array of small float32 values
    x = np.random.uniform(-0.1, 0.1, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array of float64 close to the boundaries
    x = np.array([-0.999999, 0.999999], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D complex array, complex128
    x = np.array([[0.5 - 0.5j, 2.0 + 0.0j], [-1.0j, -1.5 + 1.5j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array with only zeros, int64
    x = np.zeros((5,), dtype=np.int64)
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
