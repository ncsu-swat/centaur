
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def floor_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(3.7, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with mixed positive and negative values, float32
    x = np.array([-2.8, -0.5, 0.0, 0.5, 1.2, 3.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array, float64
    x = np.array([[-1.5, 2.3, -0.1], [4.9, -3.3, 0.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array with integer dtype int32
    x = np.array([-5, -2, 0, 1, 4, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array with float16
    x = np.array([[1.1, -2.7], [3.5, -0.9]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array, float32
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D array, float32
    x = np.random.uniform(-50.0, 50.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array with special float values (inf, -inf, nan)
    x = np.array([-np.inf, -1.5, 0.0, np.nan, 1.5, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array with int64
    x = np.array([[0, -100, 42], [7, -1, 999]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array with very small magnitudes (subnormal/close to zero)
    x = np.array([-1e-7, 1e-7, -0.999999, 0.999999], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D array, float32
    x = np.random.uniform(-5.0, 5.0, size=(1, 2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.floor_1"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_1'.")


check_valid('jax.numpy.floor', generated_inputs['jax.numpy.floor_1'], lib="jax", suffix=1)
