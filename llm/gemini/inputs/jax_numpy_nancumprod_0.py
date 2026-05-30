
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nancumprod_inputs():
    list_of_inputs = []

    # Case 1: 1D array with NaNs, axis 0, float32
    a = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D array with NaNs, axis 0, float32
    a = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D array with NaNs, axis 1, float64
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float64)
    input_dict = {"a": a, "axis": 1, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D array with NaNs, axis 2, float32
    a = np.array([[[1.0, np.nan], [3.0, 4.0]], [[np.nan, 6.0], [7.0, np.nan]]], dtype=np.float32)
    input_dict = {"a": a, "axis": 2, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 1D array with negative values, axis 0, float64
    a = np.array([-1.0, 2.0, np.nan, -4.0], dtype=np.float64)
    input_dict = {"a": a, "axis": 0, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 2D array with negative values and NaNs, axis 1, float32
    a = np.array([[-2.0, np.nan, -3.0], [np.nan, -1.0, 5.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 3D array, axis 0, float64
    a = np.random.randn(2, 3, 4).astype(np.float64)
    a[a > 1.0] = np.nan
    input_dict = {"a": a, "axis": 0, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: 4D array, axis 3, float32
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[a > 0.5] = np.nan
    input_dict = {"a": a, "axis": 3, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: 2D array, negative axis, float64
    a = np.array([[np.nan, 2.0], [3.0, np.nan]], dtype=np.float64)
    input_dict = {"a": a, "axis": -1, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 1D array, axis -1, float32
    a = np.array([np.nan, np.nan, 2.0], dtype=np.float32)
    input_dict = {"a": a, "axis": -1, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nancumprod"] = nancumprod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nancumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nancumprod'.")


check_valid('jax.numpy.nancumprod', generated_inputs['jax.numpy.nancumprod'], lib="jax", suffix=0)
