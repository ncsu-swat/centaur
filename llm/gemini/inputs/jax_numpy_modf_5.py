
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def modf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative floats
    x = np.array([1.5, -2.7, 3.0, -0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 2: 1D float64 array with integers and fractional numbers
    x = np.array([0.0, -0.0, 1.25, -3.75, 42.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 3: 2D float32 array (2x2)
    x = np.array([[1.2, 2.3], [-3.4, -4.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 4: 1D float32 array with a single element
    x = np.array([4.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 5: 2D float32 array (2x3)
    x = np.array([[0.1, -0.9, 1.5], [-2.3, 3.7, -4.1]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 6: 1D int32 array (promoted to inexact)
    x = np.array([0, 1, -1, 42, -100], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 7: 1D float32 array with large and small magnitude floats
    x = np.array([1e-5, 1e5, -1e-5, -1e5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 8: 3D float32 array (2x2x2)
    x = np.array(
        [[[0.5, -0.5], [1.5, -1.5]], [[2.5, -2.5], [3.5, -3.5]]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 9: 1D float64 array close to integer boundaries
    x = np.array([2.0001, -2.0001, 3.9999, -3.9999], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 10: 1D float32 array with special floating point values
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 11: 2D int64 array
    x = np.array([[10, -20], [30, -40]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    return list_of_inputs


generated_inputs["jax.numpy.modf_5"] = modf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.modf_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.modf_5'.")


check_valid('jax.numpy.modf', generated_inputs['jax.numpy.modf_5'], lib="jax", suffix=5)
