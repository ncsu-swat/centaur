
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 1.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 arrays with negative values
    x = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    y = np.array([-2.0, -0.1, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 arrays of same shape
    x = np.random.randn(3, 4).astype(np.float64)
    y = np.random.randn(3, 4).astype(np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D arrays (scalars) of int32
    x = np.array(5, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Same-rank broadcasting with (2, 3) and (1, 3) int32 arrays
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[2, 2, 2]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 arrays with negative and positive values
    x = np.array([-10, 20, -30, 40], dtype=np.int32)
    y = np.array([-20, 10, -40, 50], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D same-rank broadcasting (2, 2) and (2, 1) float32 arrays
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[1.5], [3.5]], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 arrays
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Same-rank broadcasting with (1, 5) and (5, 1) float32 arrays
    x = np.random.randn(1, 5).astype(np.float32)
    y = np.random.randn(5, 1).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int32 arrays of different matching values
    x = np.array([100, 200, 300], dtype=np.int32)
    y = np.array([150, 150, 150], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: One scalar (0D) and one 2D array
    x = np.random.randn(2, 2).astype(np.float32)
    y = np.array(0.0, dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gt"] = gt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gt'.")


check_valid('jax.lax.gt', generated_inputs['jax.lax.gt'], lib="jax", suffix=0)
