
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def block_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    list_of_inputs.append({"arrays": np.array([1.0, 2.0, 3.0], dtype=np.float32)})

    # Input 2: 2D float32 array
    list_of_inputs.append({"arrays": np.ones((2, 3), dtype=np.float32)})

    # Input 3: 3D int32 array
    list_of_inputs.append({"arrays": np.zeros((2, 2, 2), dtype=np.int32)})

    # Input 4: 4D float64 array
    list_of_inputs.append({"arrays": np.random.randn(1, 2, 2, 1).astype(np.float64)})

    # Input 5: 2D boolean array
    list_of_inputs.append({"arrays": np.array([[True, False], [False, True]], dtype=bool)})

    # Input 6: 1D complex64 array
    list_of_inputs.append({"arrays": np.array([1+2j, 3+4j], dtype=np.complex64)})

    # Input 7: 3D array with negative values
    list_of_inputs.append({"arrays": np.array([[[-1, -2], [-3, -4]]], dtype=np.int32)})

    # Input 8: 2D int64 array
    list_of_inputs.append({"arrays": np.arange(6, dtype=np.int64).reshape((2, 3))})

    # Input 9: 5D float32 array
    list_of_inputs.append({"arrays": np.ones((1, 1, 2, 2, 1), dtype=np.float32)})

    # Input 10: 2D complex128 array
    list_of_inputs.append({"arrays": np.array([[1j, 2j], [3j, 4j]], dtype=np.complex128)})

    return list_of_inputs

generated_inputs["jax.numpy.block_2"] = block_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.block_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.block_2'.")


check_valid('jax.numpy.block', generated_inputs['jax.numpy.block_2'], lib="jax", suffix=2)
