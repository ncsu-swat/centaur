
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def clz_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive, negative, and zero
    x = np.array([0, 1, -1, 2, 16, -16, 1024], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D int32 array
    x = np.array([[0, 1, 2], [1073741824, 65536, 255]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar int64 array
    x = np.array(42, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D int32 array
    x = np.array(
        [[[-32768, -1], [0, 1]], [[256, 1024], [16384, 32767]]],
        dtype=np.int32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D int64 array powers of 2
    x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D int32 array
    x = np.arange(-8, 8, dtype=np.int32).reshape((2, 2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D int64 array with extreme values
    x = np.array(
        [[0, 1], [np.iinfo(np.int64).min, np.iinfo(np.int64).max]],
        dtype=np.int64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D int32 array with extreme values
    x = np.array(
        [np.iinfo(np.int32).min, -1, 0, 1, np.iinfo(np.int32).max],
        dtype=np.int32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty 2D int32 array
    x = np.empty((0, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 5D int64 array
    x = np.ones((1, 2, 1, 3, 2), dtype=np.int64) * 128
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.clz"] = clz_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.clz' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.clz'.")


check_valid('jax.lax.clz', generated_inputs['jax.lax.clz'], lib="jax", suffix=0)
