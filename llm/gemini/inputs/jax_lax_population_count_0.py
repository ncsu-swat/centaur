
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def population_count_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive values
    x = np.array([0, 1, 2, 3, 4, 7, 8, 15, 16], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D int32 array with negative values
    x = np.array([-1, -2, -3, -128, -256], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar int32
    x = np.array(255, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D int32 array
    x = np.array([[0, 65535], [256, 511]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D int32 array covering boundary values
    x = np.array([[-2147483648, 2147483647], [0, -1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D int32 array
    x = np.arange(-12, 12, dtype=np.int32).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D int64 array with large values
    x = np.array(
        [
            0,
            (1 << 62) - 1,
            -1,
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
        ],
        dtype=np.int64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D int32 array
    x = np.ones((2, 2, 2, 2), dtype=np.int32) * 42
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D int64 array
    x = np.arange(10, dtype=np.int64).reshape((5, 2))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D empty int32 array
    x = np.empty((0, 5), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D int64 array with powers of two
    x = (2 ** np.arange(8, dtype=np.int64)).reshape((2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.population_count"] = population_count_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.population_count' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.population_count'.")


check_valid('jax.lax.population_count', generated_inputs['jax.lax.population_count'], lib="jax", suffix=0)
