
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def greater_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x = np.array([1.5, -2.0, 3.0, 0.0], dtype=np.float32)
    y = np.array([1.5, -1.0, 2.5, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D int32 arrays with negative values
    x = np.array([-10, 0, 10, -5], dtype=np.int32)
    y = np.array([-5, 0, 5, -10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D float64 with 1D broadcasting
    x = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float64)
    y = np.array([2.0, -2.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D arrays (scalars wrapped in ndarray)
    x = np.array(42, dtype=np.int64)
    y = np.array(-5, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D float32 with multi-axis broadcasting
    x = np.ones((2, 1, 4), dtype=np.float32)
    y = np.zeros((1, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: Boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: uint8 arrays
    x = np.array([[255, 0], [128, 64]], dtype=np.uint8)
    y = np.array([[255, 1], [127, 64]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 4D int64 arrays
    x = np.random.randint(-100, 100, size=(2, 2, 3, 3), dtype=np.int64)
    y = np.random.randint(-100, 100, size=(2, 2, 3, 3), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Float arrays with special values (infinities)
    x = np.array([-np.inf, 0.0, np.inf, 1.0], dtype=np.float32)
    y = np.array([0.0, 0.0, np.inf, -np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D broadcasting column and row vectors
    x = np.arange(5, dtype=np.int32).reshape(5, 1)
    y = np.arange(4, dtype=np.int32).reshape(1, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: float16 arrays
    x = np.array([-0.5, 1.25, 3.75], dtype=np.float16)
    y = np.array([0.0, 1.25, 2.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

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
