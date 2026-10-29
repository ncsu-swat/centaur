
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nancumsum_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs, positive axis
    a = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with NaNs, axis 0
    a = np.array(
        [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float64
    )
    input_dict = {"a": a, "axis": 0, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with negative values and NaNs, axis 1
    a = np.array(
        [[-2.5, np.nan, 1.2], [np.nan, -0.5, -3.1]], dtype=np.float32
    )
    input_dict = {"a": a, "axis": 1, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array with negative axis
    a = np.array(
        [[np.nan, 2.0], [3.0, np.nan], [-1.0, 4.0]], dtype=np.float64
    )
    input_dict = {"a": a, "axis": -1, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with NaNs, axis 0
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 0, 0] = np.nan
    input_dict = {"a": a, "axis": 0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis 2
    a = np.random.randn(3, 2, 5).astype(np.float64)
    a[1, :, 2] = np.nan
    input_dict = {"a": a, "axis": 2, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with negative axis -2
    a = np.random.randn(2, 4, 3).astype(np.float32)
    a[:, 2, :] = np.nan
    input_dict = {"a": a, "axis": -2, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, axis 1
    a = np.random.randn(2, 3, 2, 2).astype(np.float32)
    a[0, :, 1, 0] = np.nan
    input_dict = {"a": a, "axis": 1, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with negative axis
    a = np.array([-10.0, np.nan, 20.0, -30.0], dtype=np.float64)
    input_dict = {"a": a, "axis": -1, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array of all NaNs
    a = np.full((3, 3), np.nan, dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Converting float array to int32 dtype
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nancumsum"] = nancumsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nancumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nancumsum'.")


check_valid('jax.numpy.nancumsum', generated_inputs['jax.numpy.nancumsum'], lib="jax", suffix=0)
