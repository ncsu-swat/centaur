
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def abs_inputs():
    list_of_inputs = []

    # Input 1: 1D Float32 array with mixed signs
    x = np.array([-1.5, 0.0, 2.5, -3.14], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D Float64 array with mixed signs
    x = np.array([[-2.0, 3.5], [-1.2, -0.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D Signed Integer (Int32) array
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Complex64 elements
    x = np.array([3 + 4j, -5 + 12j, -1j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D Float32 array
    x = np.random.uniform(-10, 10, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (Scalar-like tensor)
    x = np.array(-42.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D Int8 array
    x = np.array([-128, -1, 0, 127], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D Complex128 array
    x = np.array(
        [[1 - 1j, -2 + 2j], [3.5 - 4.5j, -0.1 + 0.1j]], dtype=np.complex128
    )
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D Float32 array
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D Int16 array
    x = np.array([-32768, -100, 0, 100, 32767], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

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
