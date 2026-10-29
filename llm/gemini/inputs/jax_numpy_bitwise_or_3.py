
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with True
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D int32 array with negative values and False
    x = np.array([-4, -3, -2, -1, 0, 1], dtype=np.int32)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D boolean array with True
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D boolean array with False
    x = np.array([[True, False, True], [False, False, True]], dtype=bool)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D int32 array with True
    x = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 1D int64 array with large values and True
    x = np.array([1000000000, -2000000000, 3000000000], dtype=np.int64)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 0D scalar int32 array with False
    x = np.array(42, dtype=np.int32)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 4D int32 array with True
    x = np.arange(-8, 8, dtype=np.int32).reshape((2, 2, 2, 2))
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D int64 array with False
    x = np.array([[0, 255], [65535, 16777215]], dtype=np.int64)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 3D boolean array with False
    x = np.array(
        [[[True, False], [False, False]], [[True, True], [False, True]]],
        dtype=bool,
    )
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 1D bool array with True
    x = np.array([True, False, False, True], dtype=bool)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_or_3"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_or_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_or_3'.")


check_valid('jax.numpy.bitwise_or', generated_inputs['jax.numpy.bitwise_or_3'], lib="jax", suffix=3)
