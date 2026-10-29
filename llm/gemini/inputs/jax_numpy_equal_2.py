
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def equal_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array and float 0.0
    x = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    y = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float64 array and float 1.5
    x = np.array([[1.5, 2.0], [-1.5, 1.5]], dtype=np.float64)
    y = 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D int32 array and float -2.0
    x = np.array([[[-2, 0], [1, -2]], [[3, -2], [-2, 5]]], dtype=np.int32)
    y = -2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar array and float 42.0
    x = np.array(42.0, dtype=np.float32)
    y = 42.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D array with negative zero and float -0.0
    x = np.array([-0.0, 0.0, 1.0], dtype=np.float32)
    y = -0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D float32 array and float 0.5
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    x[0, 1, 1, 1] = 0.5
    y = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 1D array containing infinities and float inf
    x = np.array([float("inf"), float("-inf"), 1.0], dtype=np.float64)
    y = float("inf")
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D array containing NaN and float nan
    x = np.array([float("nan"), 0.0, 1.0], dtype=np.float32)
    y = float("nan")
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Empty 1D float array and float 10.0
    x = np.array([], dtype=np.float32)
    y = 10.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D uint8 array and float 255.0
    x = np.array([[0, 255], [255, 128]], dtype=np.uint8)
    y = 255.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 5D float16 array and float -1.0
    x = np.zeros((1, 2, 1, 3, 2), dtype=np.float16)
    x[0, 1, 0, 2, 1] = -1.0
    y = -1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.equal_2"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_2'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_2'], lib="jax", suffix=2)
