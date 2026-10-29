
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argwhere_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, size smaller than number of nonzeros
    input_dict = {
        "a": np.array([True, False, True, True, False]),
        "size": 2,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D integer array with positive and negative values, size larger than nonzeros
    input_dict = {
        "a": np.array([0, -5, 0, 12, 0, -1, 3], dtype=np.int32),
        "size": 6,
        "fill_value": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array
    input_dict = {
        "a": np.array([[1.0, 0.0, -2.5], [0.0, 3.2, 0.0]], dtype=np.float32),
        "size": 4,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array with zero size requested
    input_dict = {
        "a": np.array([[0.0, 1.5], [-3.2, 0.0]], dtype=np.float64),
        "size": 0,
        "fill_value": np.array(0, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array
    input_dict = {
        "a": np.array(
            [[[1, 0], [0, 2]], [[-1, 0], [0, 0]]],
            dtype=np.int64,
        ),
        "size": 5,
        "fill_value": np.array(99, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D uint8 array
    input_dict = {
        "a": np.array([[0, 255, 0], [128, 0, 64]], dtype=np.uint8),
        "size": 3,
        "fill_value": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D complex64 array with nonzeros
    input_dict = {
        "a": np.array([0 + 0j, 1 + 2j, 0 - 1j, 0 + 0j], dtype=np.complex64),
        "size": 4,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array
    input_dict = {
        "a": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "size": 2,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    input_dict["a"][0, 1, 0, 1] = 1.0
    input_dict["a"][1, 0, 1, 0] = -4.0
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array of all zeros
    input_dict = {
        "a": np.zeros((3, 4), dtype=np.int32),
        "size": 3,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array of all non-zeros
    input_dict = {
        "a": np.ones((2, 3), dtype=np.int32),
        "size": 8,
        "fill_value": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D boolean array
    input_dict = {
        "a": np.array(
            [[[True, False], [False, True]], [[False, False], [True, True]]],
            dtype=bool,
        ),
        "size": 4,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.argwhere_1"] = argwhere_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argwhere_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argwhere_1'.")


check_valid('jax.numpy.argwhere', generated_inputs['jax.numpy.argwhere_1'], lib="jax", suffix=1)
