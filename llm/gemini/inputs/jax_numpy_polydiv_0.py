
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polydiv_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D arrays, trim_leading_zeros=False
    u = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    v = np.array([4.0, 1.0], dtype=np.float32)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Same as 1 but with trim_leading_zeros=True
    u = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    v = np.array([4.0, 1.0], dtype=np.float32)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative coefficients, float64
    u = np.array([-1.0, 0.0, 2.0, -5.0], dtype=np.float64)
    v = np.array([1.0, -1.0], dtype=np.float64)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative coefficients, float64, trim_leading_zeros=True
    u = np.array([-1.0, 0.0, 2.0, -5.0], dtype=np.float64)
    v = np.array([1.0, -1.0], dtype=np.float64)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer coefficients (int32)
    u = np.array([10, -3, 4, 8], dtype=np.int32)
    v = np.array([2, 1], dtype=np.int32)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer coefficients (int64), trim_leading_zeros=True
    u = np.array([1, 0, 0, -1], dtype=np.int64)
    v = np.array([1, -1], dtype=np.int64)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Degree of u < degree of v (quotient should be 0, remainder u)
    u = np.array([1.0, 2.0], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Degree of u < degree of v, trim_leading_zeros=True
    u = np.array([1.0, 2.0], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element arrays (constants)
    u = np.array([8.0], dtype=np.float32)
    v = np.array([2.0], dtype=np.float32)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger degree difference
    u = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    v = np.array([1.0, 1.0], dtype=np.float64)
    input_dict = {"u": u, "v": v, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.polydiv"] = polydiv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polydiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polydiv'.")


check_valid('jax.numpy.polydiv', generated_inputs['jax.numpy.polydiv'], lib="jax", suffix=0)
