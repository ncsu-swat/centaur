
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def stack_inputs():
    list_of_inputs = []

    # Input 1: 2D array, stack along axis 0, float32
    list_of_inputs.append({
        "arrays": np.random.randn(2, 3).astype(np.float32),
        "axis": 0,
        "dtype": np.dtype('float32')
    })

    # Input 2: 2D array, stack along axis -1, int32
    list_of_inputs.append({
        "arrays": np.random.randint(-10, 10, size=(3, 4)).astype(np.int32),
        "axis": -1,
        "dtype": np.dtype('int32')
    })

    # Input 3: 3D array, stack along axis 1, float64
    list_of_inputs.append({
        "arrays": np.random.randn(2, 2, 3).astype(np.float64),
        "axis": 1,
        "dtype": np.dtype('float64')
    })

    # Input 4: 2D array, stack along axis 0, float32
    list_of_inputs.append({
        "arrays": np.random.randn(4, 5).astype(np.float32),
        "axis": 0,
        "dtype": np.dtype('float32')
    })

    # Input 5: 3D array, stack along axis 2, int64
    list_of_inputs.append({
        "arrays": np.random.randint(-100, 100, size=(3, 2, 2)).astype(np.int64),
        "axis": 2,
        "dtype": np.dtype('int64')
    })

    # Input 6: 3D array, stack along axis -2, float32
    list_of_inputs.append({
        "arrays": np.random.randn(2, 3, 4).astype(np.float32),
        "axis": -2,
        "dtype": np.dtype('float32')
    })

    # Input 7: 2D array, stack along axis 1, float16
    list_of_inputs.append({
        "arrays": np.random.randn(5, 10).astype(np.float16),
        "axis": 1,
        "dtype": np.dtype('float16')
    })

    # Input 8: 4D array, stack along axis 3, float32
    list_of_inputs.append({
        "arrays": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "axis": 3,
        "dtype": np.dtype('float32')
    })

    # Input 9: 2D array, stack along axis 0, int8
    list_of_inputs.append({
        "arrays": np.random.randint(-128, 127, size=(10, 2)).astype(np.int8),
        "axis": 0,
        "dtype": np.dtype('int8')
    })

    # Input 10: 3D array, stack along axis -1, complex64
    list_of_inputs.append({
        "arrays": (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex64),
        "axis": -1,
        "dtype": np.dtype('complex64')
    })

    return list_of_inputs

generated_inputs["jax.numpy.stack_1"] = stack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.stack_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.stack_1'.")


check_valid('jax.numpy.stack', generated_inputs['jax.numpy.stack_1'], lib="jax", suffix=1)
