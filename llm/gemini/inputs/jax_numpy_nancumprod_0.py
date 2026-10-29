
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nancumprod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaN values, axis 0
    a = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    axis = 0
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 2: 2D float64 array with NaNs, axis 1
    a = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, -4.0]], dtype=np.float64)
    axis = 1
    dtype = np.float64
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 3: 2D array, negative axis
    a = np.array([[np.nan, -2.5], [3.0, np.nan], [0.5, 2.0]], dtype=np.float32)
    axis = -1
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 4: 2D array accumulating along axis 0
    a = np.array([[2.0, 3.0], [np.nan, -1.0], [4.0, np.nan]], dtype=np.float32)
    axis = 0
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 5: 3D float32 array with NaNs, axis 2
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    axis = 2
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 6: 3D float64 array, axis 0
    a = np.array(
        [[[np.nan, 1.5], [2.0, np.nan]], [[3.0, 4.0], [np.nan, -0.5]]],
        dtype=np.float64,
    )
    axis = 0
    dtype = np.float64
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 7: 3D array, negative axis -2
    a = np.array(
        [[[1.0, np.nan], [np.nan, 2.0]], [[-3.0, 4.0], [5.0, np.nan]]],
        dtype=np.float32,
    )
    axis = -2
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 8: 1D array with negative numbers and zeros
    a = np.array([-2.0, np.nan, 0.0, -5.0, np.nan, 3.0], dtype=np.float32)
    axis = 0
    dtype = np.float64
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 9: 4D float32 array with NaNs, axis 3
    a = np.ones((2, 2, 2, 3), dtype=np.float32)
    a[0, 0, 0, 1] = np.nan
    a[1, 1, 1, 2] = -2.0
    axis = 3
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 10: 2D array, all NaNs
    a = np.full((3, 3), np.nan, dtype=np.float32)
    axis = 1
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 11: Complex float64 array with NaNs
    a = np.array(
        [[1.0 + 2.0j, np.nan], [np.nan, -3.0 + 0.5j]], dtype=np.complex128
    )
    axis = 1
    dtype = np.complex128
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

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
