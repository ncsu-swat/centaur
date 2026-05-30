
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def glu_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0 (size 4)
    x = np.random.randn(4).astype(np.float32)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=1 (size 6)
    x = np.random.randn(3, 6).astype(np.float32)
    input_dict = {"x": x, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=0 (size 2)
    x = np.random.randn(2, 5).astype(np.float32)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=2 (size 8)
    x = np.random.randn(2, 3, 8).astype(np.float32)
    input_dict = {"x": x, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=1 (size 4)
    x = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {"x": x, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, axis=-1 (size 10)
    x = np.random.randn(2, 2, 2, 10).astype(np.float32)
    input_dict = {"x": x, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 array, 1D, axis=0 (size 8)
    x = np.random.randn(8).astype(np.float64)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: array with negative values, 2D, axis=1 (size 4)
    x = np.array([[-1.0, -2.0, 3.0, 4.0], [5.0, -6.0, -7.0, 8.0]]).astype(np.float32)
    input_dict = {"x": x, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large array, axis=0 (size 128)
    x = np.random.randn(128, 64).astype(np.float32)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, axis=-1 (size 2)
    x = np.random.randn(5, 2).astype(np.float32)
    input_dict = {"x": x, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.glu"] = glu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.glu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.glu'.")


check_valid('jax.nn.glu', generated_inputs['jax.nn.glu'], lib="jax", suffix=0)
