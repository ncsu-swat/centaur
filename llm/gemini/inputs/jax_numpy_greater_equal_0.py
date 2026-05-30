
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def greater_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([1.5, -2.0, 3.5, 0.0], dtype=np.float32)
    y = np.array([1.0, -2.0, 4.0, -0.5], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 arrays with negative values
    x = np.array([10, -5, 0, 20], dtype=np.int32)
    y = np.array([5, -5, 1, 15], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 arrays (same shape)
    x = np.random.randn(3, 3).astype(np.float64)
    y = np.random.randn(3, 3).astype(np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 arrays (same shape)
    x = np.random.randint(-10, 10, size=(2, 4)).astype(np.int64)
    y = np.random.randint(-10, 10, size=(2, 4)).astype(np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcast (2D and 1D)
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(4).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcast (3D and 1D)
    x = np.random.randn(2, 3, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D arrays (scalars represented as numpy arrays)
    x = np.array(5.0, dtype=np.float32)
    y = np.array(5.0, dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 arrays (same shape)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcast (1D and 2D)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.greater_equal"] = greater_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_equal'.")


check_valid('jax.numpy.greater_equal', generated_inputs['jax.numpy.greater_equal'], lib="jax", suffix=0)
