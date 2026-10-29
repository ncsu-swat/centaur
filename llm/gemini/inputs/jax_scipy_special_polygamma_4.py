
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polygamma_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar) n=0 (digamma), positive float x
    n = np.array(0, dtype=np.int32)
    x = 1.5
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 2: 1D array with orders [0, 1, 2], positive integer float x
    n = np.array([0, 1, 2], dtype=np.int32)
    x = 2.0
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 3: 1D array of int64, small positive float x
    n = np.array([1, 2, 3, 4], dtype=np.int64)
    x = 0.5
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 4: 2D array of orders, non-integer float x
    n = np.array([[0, 1], [2, 3]], dtype=np.int32)
    x = 3.7
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 5: 2D array of all zeros (digamma function), larger x
    n = np.zeros((3, 3), dtype=np.int64)
    x = 10.0
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 6: 3D array of ones (trigamma function), medium float x
    n = np.ones((2, 2, 2), dtype=np.int32)
    x = 4.2
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 7: 1D array, negative non-integer float x
    n = np.array([0, 0, 1, 1], dtype=np.int32)
    x = -1.5
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 8: 2D array with varied orders, small float near zero
    n = np.array([[1, 0, 2], [3, 1, 0]], dtype=np.int64)
    x = 0.1
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 9: 3D array with orders up to 4, positive float x
    n = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]], dtype=np.int32)
    x = 5.5
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 10: 4D array with shape (1, 2, 1, 3), large float x
    n = np.zeros((1, 2, 1, 3), dtype=np.int32)
    x = 100.0
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 11: 1D array of higher derivative orders, fractional x
    n = np.array([5, 6], dtype=np.int64)
    x = 1.25
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.polygamma_4"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.polygamma_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.polygamma_4'.")


check_valid('jax.scipy.special.polygamma', generated_inputs['jax.scipy.special.polygamma_4'], lib="jax", suffix=4)
