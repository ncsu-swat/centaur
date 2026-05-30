
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polymul_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D arrays, trim_leading_zeros=False
    a1 = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    a2 = np.array([5.0, 0.0, 3.0], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer arrays with negative values, trim_leading_zeros=True
    a1 = np.array([-1, 2, -3], dtype=np.int32)
    a2 = np.array([4, -5], dtype=np.int32)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 arrays, trim_leading_zeros=False
    a1 = np.array([1.5, -2.5, 3.0], dtype=np.float64)
    a2 = np.array([0.5, 2.0], dtype=np.float64)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex numbers, trim_leading_zeros=False
    a1 = np.array([2.0, 1.0 + 2.0j, 1.0 - 2.0j], dtype=np.complex64)
    a2 = np.array([0.0, 5.0, 0.0, 3.0], dtype=np.complex64)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex numbers, trim_leading_zeros=True
    a1 = np.array([2.0, 1.0 + 2.0j, 1.0 - 2.0j], dtype=np.complex64)
    a2 = np.array([0.0, 5.0, 0.0, 3.0], dtype=np.complex64)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Leading zeros in inputs, trim_leading_zeros=False
    a1 = np.array([0.0, 0.0, 3.0, 4.0], dtype=np.float32)
    a2 = np.array([0.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Leading zeros in inputs, trim_leading_zeros=True
    a1 = np.array([0.0, 0.0, 3.0, 4.0], dtype=np.float32)
    a2 = np.array([0.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element arrays (scalars as 1D arrays), trim_leading_zeros=False
    a1 = np.array([5.0], dtype=np.float32)
    a2 = np.array([10.0], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large random arrays, trim_leading_zeros=False
    a1 = np.random.randn(50).astype(np.float32)
    a2 = np.random.randn(50).astype(np.float32)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int64 arrays, trim_leading_zeros=True
    a1 = np.array([1, 0, 0, 2], dtype=np.int64)
    a2 = np.array([0, 0, 3, 4], dtype=np.int64)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: All zeros, trim_leading_zeros=True
    a1 = np.array([0.0, 0.0], dtype=np.float32)
    a2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "trim_leading_zeros": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.polymul"] = polymul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polymul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polymul'.")


check_valid('jax.numpy.polymul', generated_inputs['jax.numpy.polymul'], lib="jax", suffix=0)
