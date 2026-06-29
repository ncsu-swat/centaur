
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 1D array, expand at start
    array = np.random.randn(5).astype(np.float32)
    dimensions = (0,)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 2: 1D array, expand at end
    array = np.random.randn(5).astype(np.float32)
    dimensions = (1,)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 3: 2D array, insert multiple dimensions
    array = np.random.randn(2, 3).astype(np.float32)
    dimensions = (0, 3)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 4: 2D array, insert multiple dimensions adjacent
    array = np.random.randn(2, 3).astype(np.float64)
    dimensions = (1, 2)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 5: Scalar (0D array), expand to 1D
    array = np.array(42.0).astype(np.int32)
    dimensions = (0,)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 6: 2D array, negative dimension index
    array = np.random.randn(4, 4).astype(np.int64)
    dimensions = (-1,)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 7: 2D array, multiple negative dimension indices
    array = np.random.choice([True, False], size=(2, 2)).astype(np.bool_)
    dimensions = (-3, -1)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 8: 3D array, insert multiple dimensions
    array = np.random.randn(5, 5, 5).astype(np.float32)
    dimensions = (1, 3)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 9: 1D array, insert many dimensions (complex)
    array = (np.random.randn(2) + 1j * np.random.randn(2)).astype(np.complex64)
    dimensions = (0, 1, 3)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 10: 2D array, mix of positive and negative indices
    array = np.random.randn(3, 2).astype(np.float32)
    dimensions = (0, -1)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 11: 3D array, highly sparse dimension expansion
    array = np.random.randn(1, 2, 3).astype(np.float32)
    dimensions = (0, 2, 4, 6)
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    return list_of_inputs

generated_inputs["jax.lax.expand_dims_2"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.expand_dims_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.expand_dims_2'.")


check_valid('jax.lax.expand_dims', generated_inputs['jax.lax.expand_dims_2'], lib="jax", suffix=2)
