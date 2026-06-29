
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def linalg_norm_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 vector, L1 norm, reduction over axis 0
    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    input_dict = {"x": x, "ord": 1, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 matrix, L2 norm along axis 1, keeping dimensions
    x = np.random.randn(3, 4).astype(np.float64)
    input_dict = {"x": x, "ord": 2, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 matrix, L0 norm (sparsity) along axis 0
    x = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.int32)
    input_dict = {"x": x, "ord": 0, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 tensor, L1 norm along the last axis, keeping dimensions
    x = np.random.uniform(-10, 10, (2, 3, 4)).astype(np.float32)
    input_dict = {"x": x, "ord": 1, "axis": -1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 matrix with negative values, ord=-1 along axis 0
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"x": x, "ord": -1, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float64 tensor, L2 norm along axis 2, keeping dimensions
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    input_dict = {"x": x, "ord": 2, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 vector with zeros, ord=0, keeping dimensions
    x = np.array([0.0, 1.5, 0.0, -2.5, 3.0], dtype=np.float32)
    input_dict = {"x": x, "ord": 0, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 tensor, L3 norm along axis 1
    x = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {"x": x, "ord": 3, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 matrix, ord=-2 along axis 1
    x = np.random.uniform(1, 5, (4, 4)).astype(np.float64)
    input_dict = {"x": x, "ord": -2, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 vector, L4 norm, keeping dimensions
    x = np.array([2.0, -2.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "ord": 4, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.norm_1"] = linalg_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_1'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_1'], lib="jax", suffix=1)
