
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def concatenate_inputs():
    list_of_inputs = []

    # Input 1: 2D array (unstacks to 1D arrays), axis 0
    list_of_inputs.append({
        "arrays": np.random.randn(2, 5).astype(np.float32),
        "axis": 0,
        "dtype": np.float32
    })

    # Input 2: 3D array (unstacks to 2D arrays), axis 0
    list_of_inputs.append({
        "arrays": np.random.randn(3, 4, 5).astype(np.float32),
        "axis": 0,
        "dtype": np.float32
    })

    # Input 3: 3D array (unstacks to 2D arrays), axis 1
    list_of_inputs.append({
        "arrays": np.random.randn(3, 4, 5).astype(np.float32),
        "axis": 1,
        "dtype": np.float32
    })

    # Input 4: 3D array, negative axis -1, float64
    list_of_inputs.append({
        "arrays": np.random.randn(4, 2, 3).astype(np.float64),
        "axis": -1,
        "dtype": np.float64
    })

    # Input 5: 3D array, int32, axis 1
    list_of_inputs.append({
        "arrays": np.random.randint(0, 10, (2, 3, 3)).astype(np.int32),
        "axis": 1,
        "dtype": np.int32
    })

    # Input 6: 2D array, int64, negative axis -1
    list_of_inputs.append({
        "arrays": np.random.randint(0, 10, (5, 2)).astype(np.int64),
        "axis": -1,
        "dtype": np.int64
    })

    # Input 7: 4D array, axis 1
    list_of_inputs.append({
        "arrays": np.random.randn(2, 2, 3, 4).astype(np.float32),
        "axis": 1,
        "dtype": np.float32
    })

    # Input 8: 3D array, uint8, axis 0
    list_of_inputs.append({
        "arrays": np.random.randint(0, 256, (3, 5, 5)).astype(np.uint8),
        "axis": 0,
        "dtype": np.uint8
    })

    # Input 9: 3D array, float16, negative axis -2
    list_of_inputs.append({
        "arrays": np.random.randn(2, 3, 4).astype(np.float16),
        "axis": -2,
        "dtype": np.float16
    })

    # Input 10: 5D array, axis 2
    list_of_inputs.append({
        "arrays": np.random.randn(2, 2, 2, 2, 2).astype(np.float32),
        "axis": 2,
        "dtype": np.float32
    })

    return list_of_inputs

generated_inputs["jax.numpy.concatenate_1"] = concatenate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.concatenate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.concatenate_1'.")


check_valid('jax.numpy.concatenate', generated_inputs['jax.numpy.concatenate_1'], lib="jax", suffix=1)
