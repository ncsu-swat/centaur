
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def squeeze_inputs():
    list_of_inputs = []

    # Input 1: Simple 3D array with dimension 0 of size 1, float32
    array = np.random.randn(1, 3, 4).astype(np.float32)
    dimensions = (0,)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple 3D array with dimension 1 of size 1, int32
    array = np.random.randint(0, 10, size=(3, 1, 4)).astype(np.int32)
    dimensions = (1,)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple squeeze dimensions (0, 1) in a 4D array, float64
    array = np.random.randn(1, 1, 2, 3).astype(np.float64)
    dimensions = (0, 1)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Squeezing all dimensions from a 3D array of ones, float32
    array = np.ones((1, 1, 1)).astype(np.float32)
    dimensions = (0, 1, 2)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Squeezing non-contiguous dimensions (1, 3) in a 4D array, boolean
    array = (np.random.randn(2, 1, 3, 1) > 0).astype(bool)
    dimensions = (1, 3)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative dimension index, squeezing dimension -2
    array = np.random.randn(1, 5).astype(np.float32)
    dimensions = (-2,)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative dimension index, squeezing dimension -1
    array = np.random.randn(3, 1).astype(np.float32)
    dimensions = (-1,)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High dimensional array with multiple squeeze dimensions, complex64
    array = (np.random.randn(1, 2, 1, 3, 1) + 1j * np.random.randn(1, 2, 1, 3, 1)).astype(np.complex64)
    dimensions = (0, 2, 4)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array to 0D scalar, float32
    array = np.array([42.0]).astype(np.float32)
    dimensions = (0,)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Squeezing last dimension in a 3D array, int16
    array = np.random.randint(-100, 100, size=(2, 3, 1)).astype(np.int16)
    dimensions = (2,)
    input_dict = {"array": array, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.squeeze_1"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.squeeze_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.squeeze_1'.")


check_valid('jax.lax.squeeze', generated_inputs['jax.lax.squeeze_1'], lib="jax", suffix=1)
