
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def iterable_inputs():
    list_of_inputs = []

    # Input 1: 0-dimensional scalar array (float32)
    y = np.array(3.14, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 2: 0-dimensional scalar array (int64)
    y = np.array(-42, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 3: 1-dimensional array with positive and negative integers
    y = np.array([-1, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 4: 1-dimensional empty array
    y = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 5: 2-dimensional float array
    y = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 6: 2-dimensional boolean array
    y = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 7: 3-dimensional array
    y = np.random.randn(2, 3, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 8: Complex 1-dimensional array
    y = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 9: 4-dimensional array
    y = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 10: 1-dimensional array with special values (NaN, inf)
    y = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    # Input 11: 5-dimensional array
    y = np.ones((1, 2, 1, 3, 1), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.iterable_1"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_1'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_1'], lib="jax", suffix=1)
