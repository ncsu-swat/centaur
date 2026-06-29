
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def norm_inputs():
    list_of_inputs = []

    # Input 1: 1D vector, 2-norm
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "ord": 2.0, "axis": (0,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D vector norm along axis 1, 1-norm, keepdims=True
    x = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"x": x, "ord": 1.0, "axis": (1,), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D vector norm along axis 0, inf-norm
    x = np.random.randn(2, 2).astype(np.float64)
    input_dict = {"x": x, "ord": float('inf'), "axis": (0,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrix 2-norm
    x = np.random.randn(4, 4).astype(np.float32)
    input_dict = {"x": x, "ord": 2.0, "axis": (0, 1), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matrix 1-norm with keepdims
    x = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"x": x, "ord": 1.0, "axis": (0, 1), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor vector norm along axis 2
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "ord": 2.0, "axis": (2,), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Matrix negative 1-norm
    x = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"x": x, "ord": -1.0, "axis": (0, 1), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix negative 2-norm with keepdims
    x = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"x": x, "ord": -2.0, "axis": (0, 1), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D vector 0-norm
    x = np.random.randn(5).astype(np.float32)
    input_dict = {"x": x, "ord": 0.0, "axis": (0,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D tensor vector 3-norm
    x = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {"x": x, "ord": 3.0, "axis": (1,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.norm_6"] = norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_6'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_6'], lib="jax", suffix=6)
