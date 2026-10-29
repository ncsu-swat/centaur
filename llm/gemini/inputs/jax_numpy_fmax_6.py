
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmax_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, True
    x1 = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    x2 = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D float array with negative numbers, False
    x1 = np.array([-10.5, -0.0, 0.0, 5.25], dtype=np.float32)
    x2 = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float array containing NaN and Inf, True
    x1 = np.array([[np.nan, np.inf], [-np.inf, 0.5]], dtype=np.float32)
    x2 = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D integer array, False
    x1 = np.array([[1, 2, 3], [-3, -2, -1]], dtype=np.int64)
    x2 = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float array with special values, True
    x1 = np.array(
        [[[np.nan, 2.0], [1.0, -1.0]], [[-np.inf, np.inf], [0.0, 3.5]]],
        dtype=np.float64,
    )
    x2 = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalar array, False
    x1 = np.array(42.0, dtype=np.float32)
    x2 = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 1D boolean array, True
    x1 = np.array([True, False, True, False], dtype=np.bool_)
    x2 = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 4D float array, False
    x1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x2 = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D uint8 array, True
    x1 = np.array([[0, 1, 2], [253, 254, 255]], dtype=np.uint8)
    x2 = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 1D float16 array with large dynamic range, False
    x1 = np.array([-1e4, -1.0, 0.0, 1.0, 1e4], dtype=np.float16)
    x2 = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 3D int16 array with negative values, True
    x1 = np.zeros((2, 2, 2), dtype=np.int16) - 5
    x2 = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.fmax_6"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_6'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_6'], lib="jax", suffix=6)
