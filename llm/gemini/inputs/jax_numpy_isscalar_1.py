
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isscalar_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 array (scalar)
    list_of_inputs.append({"element": np.array(1.0, dtype=np.float32)})

    # Input 2: 0D int32 array (scalar)
    list_of_inputs.append({"element": np.array(-5, dtype=np.int32)})

    # Input 3: 0D float64 array (scalar)
    list_of_inputs.append({"element": np.array(3.14159, dtype=np.float64)})

    # Input 4: 0D bool array (scalar)
    list_of_inputs.append({"element": np.array(True, dtype=bool)})

    # Input 5: 0D complex64 array (scalar)
    list_of_inputs.append({"element": np.array(1 + 2j, dtype=np.complex64)})

    # Input 6: 1D array with single element
    list_of_inputs.append({"element": np.array([1.0], dtype=np.float32)})

    # Input 7: 1D array with multiple elements
    list_of_inputs.append({"element": np.array([1, 2, 3], dtype=np.int32)})

    # Input 8: 2D array
    list_of_inputs.append({"element": np.random.randn(2, 3).astype(np.float32)})

    # Input 9: 3D array with shape (1, 1, 1)
    list_of_inputs.append({"element": np.zeros((1, 1, 1), dtype=np.float32)})

    # Input 10: Empty 1D array
    list_of_inputs.append({"element": np.array([], dtype=np.int32)})

    # Input 11: 0D uint8 array (scalar)
    list_of_inputs.append({"element": np.array(255, dtype=np.uint8)})

    return list_of_inputs

generated_inputs["jax.numpy.isscalar_1"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_1'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_1'], lib="jax", suffix=1)
