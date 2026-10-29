
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def abs_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative and positive values
    x = np.array([-3.5, -1.0, 0.0, 2.3, 4.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array
    x = np.array([[-1.2, 2.5, -3.7], [4.1, -5.0, 0.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar tensor (signed int32)
    x = np.array(-42, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D signed int64 array with negative values
    x = np.array([-100, -50, 0, 50, 100], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D signed int16 array
    x = np.array([[-32767, -1], [0, 32767]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D signed int8 array
    x = np.array([-127, -64, 0, 64, 127], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D complex64 array
    x = np.array([3.0 + 4.0j, -1.0 - 1.0j, 0.0 + 2.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D complex128 array
    x = np.array(
        [[1.5 - 2.5j, -3.0 + 0.0j], [0.0 - 4.2j, -5.5 - 1.1j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 3D float32 array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 4D float16 array
    x = (np.random.randn(2, 2, 3, 3) * 10).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array with special float values (inf, -inf, nan)
    x = np.array([-np.inf, np.inf, np.nan, -0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.abs"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.abs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.abs'.")


check_valid('jax.lax.abs', generated_inputs['jax.lax.abs'], lib="jax", suffix=0)
