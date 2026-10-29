
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def absolute_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers with positive, negative, and zero
    x = np.array([5, -2, 0, 12, -99], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array of floats
    x = np.array([[-1.5, 2.3, -0.0], [4.2, -5.6, 7.8]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Complex 1D array
    x = np.array([3 + 4j, -5 + 12j, 0 - 2j, 1 + 0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D array (scalar tensor)
    x = np.array(-42.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array of int64
    x = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array of float64
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array with special float values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, -0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex 2D array (complex128)
    x = np.array(
        [[1.0 - 1.0j, -2.5 + 3.5j], [0.0 + 0.0j, -4.0 - 4.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Boolean array
    x = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array of float16
    x = np.array([-3.14, 2.718, -1.414, 0.577], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.absolute"] = absolute_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.absolute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.absolute'.")


check_valid('jax.numpy.absolute', generated_inputs['jax.numpy.absolute'], lib="jax", suffix=0)
