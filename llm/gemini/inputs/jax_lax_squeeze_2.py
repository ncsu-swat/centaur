
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def squeeze_inputs():
    list_of_inputs = []

    # Input 1: Simple 3D array, squeeze 1st dimension
    array = np.random.randn(1, 4, 5).astype(np.float32)
    dimensions = [0]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 2: Simple 3D array, squeeze 2nd dimension
    array = np.random.randn(3, 1, 5).astype(np.float32)
    dimensions = [1]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 3: Simple 3D array, squeeze 3rd dimension with negative index
    array = np.random.randn(3, 4, 1).astype(np.float32)
    dimensions = [-1]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 4: Squeeze multiple dimensions
    array = np.random.randn(1, 2, 1, 4).astype(np.float32)
    dimensions = [0, 2]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 5: Squeeze all dimensions of a 1x1x1 array
    array = np.random.randn(1, 1, 1).astype(np.float32)
    dimensions = [0, 1, 2]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 6: float64 array, single dimension squeeze
    array = np.random.randn(5, 1).astype(np.float64)
    dimensions = [1]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 7: int32 array, squeeze multiple dimensions with negative indexes
    array = np.random.randint(0, 10, size=(1, 3, 1)).astype(np.int32)
    dimensions = [0, -1]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 8: bool array, complex squeeze
    array = (np.random.randn(1, 1, 2, 1, 3) > 0).astype(np.bool_)
    dimensions = [0, 1, 3]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 9: Large number of dimensions
    array = np.random.randn(1, 2, 1, 2, 1, 2, 1).astype(np.float32)
    dimensions = [0, 2, 4, 6]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 10: Squeeze a single dimension in a high-dimensional int64 array
    array = np.random.randint(-100, 100, size=(10, 1, 10, 10)).astype(np.int64)
    dimensions = [1]
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    # Input 11: Empty list of dimensions (valid, squeeze nothing)
    array = np.random.randn(1, 3, 1).astype(np.float32)
    dimensions = []
    list_of_inputs.append({"array": array, "dimensions": dimensions})

    return list_of_inputs

generated_inputs["jax.lax.squeeze_2"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.squeeze_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.squeeze_2'.")


check_valid('jax.lax.squeeze', generated_inputs['jax.lax.squeeze_2'], lib="jax", suffix=2)
