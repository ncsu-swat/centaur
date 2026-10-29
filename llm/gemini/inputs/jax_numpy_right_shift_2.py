
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def right_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, positive shift
    x1 = np.array([1, 2, 4, 8, 16], dtype=np.int32)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D int32 array, shift by 2
    x1 = np.array([[8, 16], [32, 64]], dtype=np.int32)
    x2 = 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 0D scalar array int64, shift by 3
    x1 = np.array(128, dtype=np.int64)
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 3D int32 array, shift by 4
    x1 = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    x2 = 4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 2D int32 array with negative values, shift by 1
    x1 = np.array([[-16, -8], [-4, -2]], dtype=np.int32)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 1D int64 array, shift by 0
    x1 = np.array([0, 10, 100, 1000], dtype=np.int64)
    x2 = 0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D int32 array, shift by 2
    x1 = np.random.randint(0, 256, size=(2, 2, 3, 3), dtype=np.int32)
    x2 = 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 2D int64 array, shift by 8
    x1 = np.array([[256, 1024], [4096, 65536]], dtype=np.int64)
    x2 = 8
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 1D int32 array with mixed sign values, shift by 1
    x1 = np.array([12, 24, -30, 60], dtype=np.int32)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 3D int64 array, shift by 5
    x1 = np.random.randint(-1000, 1000, size=(2, 2, 2), dtype=np.int64)
    x2 = 5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 1D int64 array, shift by 6
    x1 = np.array([128, 256, 512, 1024], dtype=np.int64)
    x2 = 6
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.right_shift_2"] = right_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.right_shift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.right_shift_2'.")


check_valid('jax.numpy.right_shift', generated_inputs['jax.numpy.right_shift_2'], lib="jax", suffix=2)
