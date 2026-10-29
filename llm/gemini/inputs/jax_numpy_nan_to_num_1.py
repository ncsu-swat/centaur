
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nan_to_num_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with nan, posinf, neginf
    x = np.array([0.0, np.nan, 1.0, np.inf, 2.0, -np.inf], dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 0.0,
        "posinf": 999.0,
        "neginf": -999.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array with inf and nan
    x = np.array(
        [[np.nan, 1.5, np.inf], [-np.inf, -2.5, 0.0]], dtype=np.float64
    )
    input_dict = {
        "x": x,
        "copy": False,
        "nan": -1.0,
        "posinf": 100.0,
        "neginf": -100.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D scalar float32 array with NaN
    x = np.array(np.nan, dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 42.0,
        "posinf": 1e6,
        "neginf": -1e6,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array
    x = np.array(
        [
            [[np.nan, np.inf], [1.0, 2.0]],
            [[-np.inf, 3.0], [np.nan, -np.inf]],
        ],
        dtype=np.float32,
    )
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 0.5,
        "posinf": 10.0,
        "neginf": -10.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float16 array
    x = np.array(
        [np.inf, -np.inf, np.nan, 5.0, -5.0, 0.0], dtype=np.float16
    )
    input_dict = {
        "x": x,
        "copy": False,
        "nan": 0.0,
        "posinf": 65504.0,
        "neginf": -65504.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array without special values (all finite)
    x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 0.0,
        "posinf": 1.0,
        "neginf": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array with random values and inserted NaNs/infs
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    x[0, 0, 0, 0] = np.nan
    x[0, 1, 0, 1] = np.inf
    x[1, 0, 1, 0] = -np.inf
    input_dict = {
        "x": x,
        "copy": True,
        "nan": -99.0,
        "posinf": 1000.0,
        "neginf": -1000.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array of only NaNs
    x = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    input_dict = {
        "x": x,
        "copy": False,
        "nan": 3.14159,
        "posinf": 2.71828,
        "neginf": -2.71828,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array of only positive and negative infinities
    x = np.array([np.inf, -np.inf, np.inf, -np.inf], dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 0.0,
        "posinf": 500.5,
        "neginf": -500.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 with large values and mixed infinities
    x = np.array([[1e30, np.inf], [-np.inf, np.nan]], dtype=np.float64)
    input_dict = {
        "x": x,
        "copy": False,
        "nan": 1e-5,
        "posinf": 1e38,
        "neginf": -1e38,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nan_to_num_1"] = nan_to_num_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nan_to_num_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nan_to_num_1'.")


check_valid('jax.numpy.nan_to_num', generated_inputs['jax.numpy.nan_to_num_1'], lib="jax", suffix=1)
