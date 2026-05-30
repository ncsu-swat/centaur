
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arctan_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative, zero, and positive values
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D float64 array including infinities and NaN
    x = np.array([-np.inf, -2.0, 0.0, 2.0, np.inf, np.nan], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float32 array (matrix)
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D int32 array (integers)
    x = np.array([[[1, -2], [3, 0]], [[-5, 6], [7, -8]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex64 array
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (scalar equivalent)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Large 2D float32 array
    x = np.random.uniform(-100, 100, size=(10, 10)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D float16 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D complex128 array
    x = np.array([2.0 + 7.0j, -2.0 - 7.0j], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D int64 array with large values
    x = np.array([[-1000, 0], [1000, 500]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.arctan_1"] = arctan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctan_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctan_1'.")


check_valid('jax.numpy.arctan', generated_inputs['jax.numpy.arctan_1'], lib="jax", suffix=1)
