
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, True
    x = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D bool array, False
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D int32 array with negative values, True
    x = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D bool array, True
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D int32 array, False
    x = np.arange(8, dtype=np.int32).reshape((2, 2, 2))
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 0D int64 array (scalar array), True
    x = np.array(42, dtype=np.int64)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D int32 array, False
    x = np.random.randint(-100, 100, size=(2, 3, 1, 4), dtype=np.int32)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int64 array with negative and positive values, True
    x = np.array([-1000, -1, 0, 1, 1000], dtype=np.int64)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D int64 array, True
    x = np.array([[100000, -200000], [300000, -400000]], dtype=np.int64)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 3D bool array, False
    x = np.array([[[True, False]], [[False, True]]], dtype=np.bool_)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 1D empty int32 array, True
    x = np.array([], dtype=np.int32)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_xor_4"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_4'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_4'], lib="jax", suffix=4)
