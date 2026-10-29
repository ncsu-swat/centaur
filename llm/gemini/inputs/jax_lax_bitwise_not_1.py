
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D int32 array with positive and negative integers
    x = np.array([[-10, 0, 5], [12, -1, 42]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar-like 0D int64 array
    x = np.array(-128, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D boolean array
    x = np.array(True, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D int32 array
    x = np.arange(-12, 12, dtype=np.int32).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D int64 array
    x = np.array([[100, -200], [300, -400]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D int32 array
    x = np.zeros((2, 1, 2, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D int64 array
    x = np.array([-1000, -1, 0, 1, 1000], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D boolean array
    x = np.array([[True, False], [False, False], [True, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D int32 array
    x = np.array([0, -1, 1, 255, -256], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D boolean array
    x = np.array(
        [[[True, False], [False, True]], [[False, False], [True, True]]],
        dtype=np.bool_,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.bitwise_not_1"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_not_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_not_1'.")


check_valid('jax.lax.bitwise_not', generated_inputs['jax.lax.bitwise_not_1'], lib="jax", suffix=1)
