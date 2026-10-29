
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanargmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs and negative values, float32
    a = np.array([1.0, -3.0, np.nan, 5.0, -10.0, 4.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, float64, negative axis, keepdims=True
    a = np.array([np.nan, 2.5, -1.2, 0.0, np.nan, 7.8], dtype=np.float64)
    input_dict = {"a": a, "axis": -1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=0, float32, keepdims=True
    a = np.array(
        [[np.nan, 2.0, 3.0], [4.0, np.nan, 1.0], [-5.0, 6.0, np.nan]],
        dtype=np.float32,
    )
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, axis=1, float64, keepdims=False
    a = np.array(
        [[-1.0, -5.0, np.nan], [np.nan, np.nan, 0.5], [3.2, -2.1, 4.4]],
        dtype=np.float64,
    )
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with negative axis, keepdims=False
    a = np.array(
        [[10.0, np.nan, -2.0, 8.0], [np.nan, -1.0, 3.0, np.nan]],
        dtype=np.float32,
    )
    input_dict = {"a": a, "axis": -2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=2, keepdims=True
    rng = np.random.default_rng(42)
    a = rng.standard_normal((3, 4, 5)).astype(np.float32)
    a[rng.random((3, 4, 5)) < 0.2] = np.nan
    input_dict = {"a": a, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, axis=1, keepdims=False
    a = rng.uniform(-10.0, 10.0, (2, 5, 3)).astype(np.float64)
    a[0, :, 1] = np.nan
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, negative axis=-3, keepdims=True
    a = rng.standard_normal((4, 2, 3)).astype(np.float32)
    a[1, 0, :] = np.nan
    input_dict = {"a": a, "axis": -3, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, axis=0, keepdims=False
    a = rng.standard_normal((2, 3, 2, 4)).astype(np.float32)
    a[a < 0.0] = np.nan
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, axis=3, keepdims=True, float64
    a = rng.standard_normal((3, 2, 4, 3)).astype(np.float64)
    a[0, 1, 2, :] = np.nan
    input_dict = {"a": a, "axis": 3, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array with inf and nan values
    a = np.array(
        [[np.inf, np.nan, -np.inf], [-np.inf, 100.0, np.nan]], dtype=np.float32
    )
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nanargmax"] = nanargmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanargmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanargmax'.")


check_valid('jax.numpy.nanargmax', generated_inputs['jax.numpy.nanargmax'], lib="jax", suffix=0)
