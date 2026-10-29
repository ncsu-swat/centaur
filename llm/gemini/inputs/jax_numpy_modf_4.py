
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def modf_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats (float32)
    x = np.array([1.2, 2.5, 3.7, 4.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 2: 1D array of negative floats (float32)
    x = np.array([-1.5, -2.7, -3.2, -0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 3: 2D array of mixed floats (float64)
    x = np.array([[1.25, -2.75], [3.5, -4.125]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 4: 1D array with integer values
    x = np.array([0, 1, -2, 3, -4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 5: 3D array (float32)
    x = np.array(
        [[[1.1, -2.2], [3.3, -4.4]], [[5.5, -6.6], [7.7, -8.8]]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 6: 1D single-element array (float32)
    x = np.array([4.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 7: 1D array with large/small floats (float64)
    x = np.array([1e-4, -1e-4, 1e5 + 0.5, -(1e5 + 0.5)], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 8: 2D column vector shape (3, 1)
    x = np.array([[0.125], [-1.875], [2.625]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 9: 1D array with zeros
    x = np.array([0.0, -0.0, 0.5, -0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 10: 1D array with float16
    x = np.array([1.5, -2.5, 3.25, -4.75], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    return list_of_inputs


generated_inputs["jax.numpy.modf_4"] = modf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.modf_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.modf_4'.")


check_valid('jax.numpy.modf', generated_inputs['jax.numpy.modf_4'], lib="jax", suffix=4)
