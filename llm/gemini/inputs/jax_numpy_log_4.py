
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with mathematical constants
    x = np.array([1.0, 2.7182818, 7.389056, 20.085537], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D float64 array with powers of 10
    x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D int32 array with powers of 2
    x = np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D int64 array
    x = np.array([3, 9, 27, 81, 243], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D float16 array
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D float32 array spanning small to large scales
    x = np.array([1e-6, 1e-3, 1.0, 1e3, 1e6], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Single-element 1D float32 array
    x = np.array([42.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array including zero and positive infinity
    x = np.array([0.0, 1.0, float("inf")], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Linearly spaced float32 array
    x = np.linspace(1.0, 50.0, 25, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D float64 array with fractional values between 0 and 1
    x = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.99], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D float32 array with large dynamic range
    x = np.array([1e-15, 1e-8, 1.0, 1e8, 1e15], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.log_4"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log_4'.")


check_valid('jax.numpy.log', generated_inputs['jax.numpy.log_4'], lib="jax", suffix=4)
