
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_sign_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative, zero, positive, and NaN values
    x = np.array([-5.0, -0.0, 0.0, 3.5, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float32 array
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D float64 scalar array
    x = np.array(-42.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D int32 array with negative, zero, and positive values
    x = np.array([-10, -1, 0, 1, 100], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D int64 array
    x = np.array([[-3, 0, 5], [10, -20, 0]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D int16 array
    x = np.random.randint(-50, 50, size=(2, 3, 4), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D complex64 array
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, 0.0 + 0.0j, -1.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D complex128 array
    x = np.array(
        [[2.5 - 1.5j, -4.0 + 0.0j], [0.0 - 3.0j, 1.0 + 1.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D int8 array
    x = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D float16 array
    x = np.array([-2.5, 0.0, 7.8, -0.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.sign"] = lax_sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sign'.")


check_valid('jax.lax.sign', generated_inputs['jax.lax.sign'], lib="jax", suffix=0)
