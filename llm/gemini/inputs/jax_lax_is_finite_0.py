
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def is_finite_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with finite, inf, and nan values
    x = np.array([0.0, -1.5, np.inf, -np.inf, np.nan, 3.14], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar float32 (finite)
    x = np.array(42.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar float64 (infinite)
    x = np.array(np.inf, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D float64 matrix with various values
    x = np.array(
        [[1.0, np.nan], [-np.inf, 0.0], [np.inf, -100.5]], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D float16 array
    x = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D float32 tensor
    x = np.random.randn(2, 3, 4).astype(np.float32)
    x[0, 1, 2] = np.inf
    x[1, 2, 3] = np.nan
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Empty 1D float32 array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D float32 tensor with all finite numbers
    x = np.zeros((2, 2, 3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D float64 array with extreme values
    x = np.array(
        [[1e308, -1e308], [np.finfo(np.float64).tiny, -0.0]], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D float32 array with all NaN values
    x = np.full((5,), np.nan, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D float64 array with subnormal and zero values
    x = np.array([[[0.0, -0.0], [1e-320, -1e-320]]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.is_finite"] = is_finite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.is_finite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.is_finite'.")


check_valid('jax.lax.is_finite', generated_inputs['jax.lax.is_finite'], lib="jax", suffix=0)
