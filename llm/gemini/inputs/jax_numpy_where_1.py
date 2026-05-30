
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def where_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, basic matching shapes
    condition = np.array([True, False, True])
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float32, random values
    condition = np.random.rand(3, 4) > 0.5
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D arrays, int32, matching shapes
    condition = np.random.rand(2, 2, 2) > 0.5
    x = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32)
    y = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting condition (1D) to x and y (2D)
    condition = np.array([True, False, True])
    x = np.random.randn(2, 3).astype(np.float32)
    y = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting condition (2, 1) to x and y (2, 3)
    condition = np.array([[True], [False]])
    x = np.random.randn(2, 3).astype(np.float32)
    y = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting x (1, 3) and y (2, 1) with condition (2, 3)
    condition = np.random.rand(2, 3) > 0.5
    x = np.random.randn(1, 3).astype(np.float32)
    y = np.random.randn(2, 1).astype(np.float32)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 arrays with negative and large values
    condition = np.array([False, True, False])
    x = np.array([1e10, -2e10, 3e10], dtype=np.float64)
    y = np.array([-1e10, 2e10, -3e10], dtype=np.float64)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays, float32
    condition = np.random.rand(2, 2, 2, 2) > 0.5
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    y = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 arrays, matching shapes
    condition = np.array([True, True, False])
    x = np.array([100, 200, 300], dtype=np.int64)
    y = np.array([-100, -200, -300], dtype=np.int64)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: boolean arrays for x and y
    condition = np.array([True, False, True])
    x = np.array([False, False, False], dtype=bool)
    y = np.array([True, True, True], dtype=bool)
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.where_1"] = where_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.where_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.where_1'.")


check_valid('jax.numpy.where', generated_inputs['jax.numpy.where_1'], lib="jax", suffix=1)
