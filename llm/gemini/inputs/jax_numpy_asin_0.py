
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asin_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 within [-1, 1]
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array float32
    x = np.linspace(-1.0, 1.0, num=10, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array float64
    x = np.random.uniform(-1.0, 1.0, size=(3, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array float16
    x = np.random.uniform(-0.9, 0.9, size=(2, 3, 2)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array float32
    x = np.random.uniform(-0.5, 0.5, size=(2, 2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D integer array with valid boundary values
    x = np.array([-1, 0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex64 array
    x = (
        np.random.uniform(-1.0, 1.0, size=(2, 3))
        + 1j * np.random.uniform(-1.0, 1.0, size=(2, 3))
    ).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D complex128 array
    x = np.array([-0.5 + 0.5j, 0.0 + 0.0j, 0.5 - 0.5j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D array strictly negative values in [-1, 0]
    x = np.random.uniform(-1.0, 0.0, size=(4, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D boolean array (casts naturally)
    x = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.asin"] = asin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asin'.")


check_valid('jax.numpy.asin', generated_inputs['jax.numpy.asin'], lib="jax", suffix=0)
