
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanargmin_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs, axis=0, keepdims=False
    a = np.array([np.nan, 3.0, 5.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with NaNs, axis=-1, keepdims=True
    a = np.array([-2.5, np.nan, -10.0, 4.0, np.nan], dtype=np.float64)
    input_dict = {"a": a, "axis": -1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=0, keepdims=False
    a = np.array(
        [[1.0, np.nan, 3.0], [4.0, 0.5, np.nan], [np.nan, -1.0, 2.0]],
        dtype=np.float32,
    )
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, axis=1, keepdims=True
    a = np.array(
        [[np.nan, 5.0, -3.0], [2.0, np.nan, np.nan], [7.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, axis=-1, keepdims=False, float64
    a = np.array([[-1.5, -3.2], [np.nan, -0.1], [4.4, 2.1]], dtype=np.float64)
    input_dict = {"a": a, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with all-NaN row, axis=1, keepdims=False
    a = np.array([[np.nan, np.nan], [1.0, 2.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, axis=0, keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 0, 1] = np.nan
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis=1, keepdims=False
    a = np.random.randn(4, 3, 2).astype(np.float64)
    a[a < 0] = np.nan
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, axis=2, keepdims=True
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 10.0
    a[0, 0, 0] = np.nan
    input_dict = {"a": a, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, axis=-2, keepdims=False
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    a[0, 1, :, 1] = np.nan
    input_dict = {"a": a, "axis": -2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array, axis=3, keepdims=True, float16
    a = np.array([[[[np.nan, 2.0], [3.0, 1.0]]]], dtype=np.float16)
    input_dict = {"a": a, "axis": 3, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nanargmin"] = nanargmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanargmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanargmin'.")


check_valid('jax.numpy.nanargmin', generated_inputs['jax.numpy.nanargmin'], lib="jax", suffix=0)
