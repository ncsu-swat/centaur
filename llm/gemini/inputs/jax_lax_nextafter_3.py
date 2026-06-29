
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nextafter_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, pointing towards larger value
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = 5.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, pointing towards smaller value
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = -1.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 array, pointing towards 0
    x1 = np.random.randn(3, 3).astype(np.float64)
    x2 = 0.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D array (scalar array)
    x1 = np.array(0.0, dtype=np.float32)
    x2 = 1.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, pointing towards positive infinity
    x1 = np.ones((2, 2, 2), dtype=np.float32)
    x2 = float('inf')
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float64 array, pointing towards negative infinity
    x1 = np.zeros((2, 1, 2, 1), dtype=np.float64)
    x2 = float('-inf')
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with negative values, pointing to a negative float
    x1 = np.array([-1.5, -2.5, -3.5], dtype=np.float32)
    x2 = -10.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with single element, pointing to a fractional float
    x1 = np.array([0.1], dtype=np.float32)
    x2 = 0.10001
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 5D array
    x1 = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    x2 = 100.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array containing special values, pointing to 1.0
    x1 = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    x2 = 1.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.nextafter_3"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.nextafter_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.nextafter_3'.")


check_valid('jax.lax.nextafter', generated_inputs['jax.lax.nextafter_3'], lib="jax", suffix=3)
