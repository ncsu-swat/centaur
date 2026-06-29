
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ge_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays of same shape
    x = np.array([-1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    y = np.array([-2.0, 0.0, 3.0, 1.5], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 arrays of same shape
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 2], [2, 2]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcast shapes (3, 1) and (1, 4) with float64
    x = np.array([[1.5], [-2.3], [0.0]], dtype=np.float64)
    y = np.array([[0.5, -2.3, 3.1, 0.0]], dtype=np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D broadcast shapes (1,) and (5,) with int64
    x = np.array([5], dtype=np.int64)
    y = np.array([1, 5, 10, 2, 5], dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays of same shape with float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcast shapes (1, 3) and (2, 3) with int32
    x = np.array([[10, 20, 30]], dtype=np.int32)
    y = np.array([[5, 25, 15], [15, 20, 35]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large 2D arrays of same shape with float32
    x = np.random.uniform(-100, 100, (100, 100)).astype(np.float32)
    y = np.random.uniform(-100, 100, (100, 100)).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays of same shape with int32
    x = np.random.randint(-10, 10, size=(2, 2, 2, 2)).astype(np.int32)
    y = np.random.randint(-10, 10, size=(2, 2, 2, 2)).astype(np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High dimensional broadcast (1, 5, 1) and (2, 1, 3) with float64
    x = np.random.randn(1, 5, 1).astype(np.float64)
    y = np.random.randn(2, 1, 3).astype(np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Simple 1D int64 arrays of same shape
    x = np.array([-128, 0, 127], dtype=np.int64)
    y = np.array([-128, 1, 127], dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.ge"] = ge_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.ge' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.ge'.")


check_valid('jax.lax.ge', generated_inputs['jax.lax.ge'], lib="jax", suffix=0)
