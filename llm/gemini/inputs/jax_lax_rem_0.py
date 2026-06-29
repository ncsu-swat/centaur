
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rem_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, same shape
    x = np.array([5.5, 6.2, 7.8], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, same shape with negative values
    x = np.array([[-10.5, 12.0], [15.5, -20.0]], dtype=np.float32)
    y = np.array([[3.0, -4.0], [-5.0, 6.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32, same shape
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([3, 4, 7], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int32, same shape with negative values
    x = np.array([[-15, 25], [-35, 45]], dtype=np.int32)
    y = np.array([[4, -6], [-8, 10]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64, same shape
    x = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float64)
    y = np.array([[[0.5, 1.2], [1.5, 2.2]], [[2.5, 3.2], [3.5, 4.2]]], dtype=np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64, same shape
    x = np.array([100, 200, 300, 400], dtype=np.int64)
    y = np.array([15, 25, 35, 45], dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 with broadcasting (1, 3) and (2, 3)
    x = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    y = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32 with broadcasting (3, 1) and (3, 3)
    x = np.array([[10], [20], [30]], dtype=np.int32)
    y = np.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 with broadcasting (2, 1, 2) and (2, 2, 2)
    x = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    y = np.array([[[0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 0D float32 (scalar-like arrays)
    x = np.array(15.5, dtype=np.float32)
    y = np.array(4.0, dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.rem"] = rem_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.rem' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.rem'.")


check_valid('jax.lax.rem', generated_inputs['jax.lax.rem'], lib="jax", suffix=0)
