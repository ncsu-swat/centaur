
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def less_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, positive integer x
    x = 3
    y = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D float array, negative integer x
    x = -2
    y = np.array([-5.0, -2.0, 0.0, 3.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D integer array, zero x
    x = 0
    y = np.array([[-1, 0, 1], [2, -3, 4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D float64 array, large positive integer x
    x = 100
    y = np.array([[50.5, 150.2], [99.9, 100.1]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D float32 array
    x = 5
    y = np.random.uniform(0, 10, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 0D scalar array
    x = 10
    y = np.array(15, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D int16 array with negative values
    x = -10
    y = np.random.randint(-20, 20, size=(2, 2, 3, 2), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D uint8 array
    x = 128
    y = np.array([0, 64, 128, 200, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Empty array
    x = 1
    y = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D array with inf and nan values
    x = 0
    y = np.array([[-np.inf, 0.0], [np.nan, np.inf]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 5D integer array
    x = 42
    y = np.zeros((1, 2, 1, 3, 1), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.less_5"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_5'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_5'], lib="jax", suffix=5)
