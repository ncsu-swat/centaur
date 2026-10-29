
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def signbit_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 with positive, negative, and signed zero
    x = np.array([-1.5, -0.0, 0.0, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D int32 array
    x = np.array([-10, -1, 0, 1, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float64 with special values (NaNs and infinities)
    x = np.array([[np.nan, -np.nan], [np.inf, -np.inf]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar float32 negative zero
    x = np.array(-0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 0D scalar int64
    x = np.array(42, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D boolean array
    x = np.array([True, False, True, False], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D float32 array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D int64 array with boundary and negative values
    x = np.array([[-9223372036854775808, 0], [1, -1]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D float64 array
    x = np.random.randn(1, 2, 2, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D float32 array with zeros and ones
    x = np.array([[-0.0, 0.0], [-1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D int64 array with strictly negative integers
    x = np.array([-100, -50, -1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.signbit"] = signbit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.signbit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.signbit'.")


check_valid('jax.numpy.signbit', generated_inputs['jax.numpy.signbit'], lib="jax", suffix=0)
