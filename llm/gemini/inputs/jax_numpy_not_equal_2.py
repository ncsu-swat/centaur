
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array and a zero float
    x = np.array([0.0, 1.0, -1.0, 0.0, 2.5], dtype=np.float32)
    y = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float64 array with positive and negative values
    x = np.array([[-1.5, 2.0, 3.5], [4.0, -1.5, 0.0]], dtype=np.float64)
    y = -1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar array (tensor) and a float
    x = np.array(5.0, dtype=np.float32)
    y = 5.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int32 array compared against a float
    x = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    y = 10.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D array containing infinities and NaNs
    x = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    y = float("inf")
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 2D boolean array compared against float (0.0 or 1.0)
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D float32 tensor
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int64 array with negative numbers
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    y = -5.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Empty array
    x = np.array([], dtype=np.float32)
    y = 3.14
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D float16 array
    x = np.linspace(-2.0, 2.0, 9, dtype=np.float16).reshape(3, 3)
    y = -2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D float32 array with scientific notation float
    x = np.ones((2, 4, 2), dtype=np.float32) * 1e-4
    y = 1e-4
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.not_equal_2"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_2'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_2'], lib="jax", suffix=2)
