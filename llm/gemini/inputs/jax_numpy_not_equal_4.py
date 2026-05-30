
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, matching x is 0.0
    x = 0.0
    y = np.array([0.0, 1.0, -2.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array with random values
    x = -1.5
    y = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, matching some values
    x = 2.5
    y = np.array([[2.5, 1.0, 2.5], [0.0, 2.5, -2.5]], dtype=np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array of zeros
    x = 0.0
    y = np.zeros((2, 2, 2), dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32 array, small positive float
    x = 1e-5
    y = np.array([1e-5, 2e-5, 1e-5], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array with negative values
    x = -999.9
    y = np.linspace(-1000, -999, 6).reshape(2, 3).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array filled with x
    x = 4.2
    y = np.full((2, 3, 2), 4.2, dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Infinity as float
    x = float('inf')
    y = np.array([1.0, float('inf'), float('-inf')], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float64 array
    x = 3.14159
    y = np.ones((1, 2, 2, 3), dtype=np.float64) * 3.14159
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: -0.0 and 0.0
    x = -0.0
    y = np.array([0.0, -0.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int32 array (representing tensor) with float x
    x = 5.0
    y = np.array([1, 2, 5, 6], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.not_equal_4"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_4'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_4'], lib="jax", suffix=4)
