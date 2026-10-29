
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isnan_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaN, Inf, and finite values
    x = np.array([np.nan, 1.0, -2.5, np.inf, -np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array with mixed values and NaNs
    x = np.array([[np.nan, -1.0], [0.0, np.nan]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar float32 array with NaN
    x = np.array(np.nan, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar float64 array without NaN
    x = np.array(3.14159, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float16 array
    x = np.array(
        [[[np.nan, 1.0], [-1.0, 0.0]], [[2.0, np.nan], [np.inf, -np.nan]]],
        dtype=np.float16,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D int32 array (contains no NaNs)
    x = np.array([-10, 0, 10, 100], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex64 array with NaNs in real and imaginary parts
    x = np.array(
        [[1.0 + 2.0j, np.nan + 0.0j], [0.0 + np.nan * 1j, -1.0 - 1.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D float32 array
    x = np.zeros((2, 3, 4, 5), dtype=np.float32)
    x[0, 1, 2, 3] = np.nan
    x[1, 2, 1, 0] = np.nan
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty 1D float32 array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D complex128 array with inf and nan
    x = np.array(
        [complex(np.nan, np.inf), complex(1.0, 2.0), complex(np.nan, np.nan)],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.isnan_1"] = isnan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isnan_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isnan_1'.")


check_valid('jax.numpy.isnan', generated_inputs['jax.numpy.isnan_1'], lib="jax", suffix=1)
