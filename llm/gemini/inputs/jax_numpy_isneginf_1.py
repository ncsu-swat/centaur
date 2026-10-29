
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isneginf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with -inf, inf, nan, and finite values
    x = np.array([-np.inf, 5.0, np.inf, np.nan, -1.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 2: 2D float64 array (3x3) with -inf on diagonal
    x = np.array(
        [
            [-np.inf, 1.5, 2.3],
            [0.0, -np.inf, -4.2],
            [np.nan, np.inf, -np.inf],
        ],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 3: 0D scalar float32 array with -inf
    x = np.array(-np.inf, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 4: 1D int32 array
    x = np.array([-10, -1, 0, 1, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 5: 3D float32 array (2, 2, 3)
    x = np.array(
        [
            [[-np.inf, 1.0, 2.0], [np.inf, -np.inf, 0.0]],
            [[np.nan, -3.0, -np.inf], [4.0, 5.0, 6.0]],
        ],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 6: 1D float16 array
    x = np.array([-np.inf, -100.0, 0.0, 100.0, np.inf], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 7: 2D float32 array with all -inf
    x = np.full((2, 4), -np.inf, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 8: 1D float64 array with no -inf
    x = np.array([np.nan, np.inf, 1e10, -1e10, 0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 9: 4D float32 array with shape (1, 2, 1, 3)
    x = np.array([[[[-np.inf, 0.5, np.inf]], [[-2.5, -np.inf, np.nan]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 10: 1D empty float32 array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 11: 2D int64 array
    x = np.array([[-100, 0, 100], [200, -300, 400]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    return list_of_inputs


generated_inputs["jax.numpy.isneginf_1"] = isneginf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isneginf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isneginf_1'.")


check_valid('jax.numpy.isneginf', generated_inputs['jax.numpy.isneginf_1'], lib="jax", suffix=1)
