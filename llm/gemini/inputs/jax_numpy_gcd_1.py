
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gcd_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays of same shape
    x1 = np.array([12, 18, 24, 0], dtype=np.int32)
    x2 = np.array([5, 10, 15, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D int64 arrays with negative values
    x1 = np.array([[-12, 36], [48, -60]], dtype=np.int64)
    x2 = np.array([[18, -24], [-16, 20]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 0D arrays (scalars wrapped as arrays)
    x1 = np.array(42, dtype=np.int32)
    x2 = np.array(56, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting (3, 1) and (1, 4)
    x1 = np.array([[6], [15], [28]], dtype=np.int32)
    x2 = np.array([[9, 12, 21, 35]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 1D int32 arrays
    x1 = np.array([10, 20, 30, 40], dtype=np.int32)
    x2 = np.array([15, 25, 35, 45], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 3D int32 arrays with zeros
    x1 = np.zeros((2, 3, 4), dtype=np.int32)
    x2 = np.arange(1, 25, dtype=np.int32).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Broadcasting 1D to 2D with int32
    x1 = np.array([4, 8, 12], dtype=np.int32)
    x2 = np.array([[2, 4, 6], [8, 16, 24]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D int32 arrays with coprimes
    x1 = np.array([7, 11, 13, 17, 19], dtype=np.int32)
    x2 = np.array([23, 29, 31, 37, 41], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Large int64 values
    x1 = np.array([1000000007, 2147483648], dtype=np.int64)
    x2 = np.array([1000000009, 1073741824], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 4D int32 array with 0D array broadcast
    x1 = np.ones((2, 2, 2, 2), dtype=np.int32) * 30
    x2 = np.array(45, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 2D int64 arrays with zeros and negatives
    x1 = np.array([[0, -10], [20, -30]], dtype=np.int64)
    x2 = np.array([[5, 0], [-15, -45]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.gcd_1"] = gcd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gcd_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gcd_1'.")


check_valid('jax.numpy.gcd', generated_inputs['jax.numpy.gcd_1'], lib="jax", suffix=1)
