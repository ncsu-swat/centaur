
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isfinite_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with finite, inf, -inf, nan
    x = np.array([-1.0, 0.0, 3.5, np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array with all finite values
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array (float32)
    x = np.array(np.inf, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar array with finite value
    x = np.array(42.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float16 array containing infinities and nans
    x = np.array(
        [[[np.nan, 1.0], [np.inf, -2.0]], [[-np.inf, 0.0], [5.0, np.nan]]],
        dtype=np.float16,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D integer array (always finite)
    x = np.array([-10, -1, 0, 1, 10, 100], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D int64 array
    x = np.arange(12, dtype=np.int64).reshape((4, 3))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D complex64 array with finite and infinite components
    x = np.array(
        [1.0 + 2.0j, np.inf + 1.0j, 0.0 + np.nan * 1j, -np.inf - 1j],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D complex128 array with purely finite values
    x = np.array([[1.0 - 1.0j, 2.0 + 3.0j], [-4.0j, 5.0]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 4D float32 array
    x = np.ones((2, 2, 3, 2), dtype=np.float32)
    x[0, 1, 2, 0] = np.inf
    x[1, 0, 1, 1] = np.nan
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D boolean array (always finite)
    x = np.array([True, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.isfinite_1"] = isfinite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isfinite_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isfinite_1'.")


check_valid('jax.numpy.isfinite', generated_inputs['jax.numpy.isfinite_1'], lib="jax", suffix=1)
