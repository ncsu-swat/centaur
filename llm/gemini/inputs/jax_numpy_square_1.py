
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def square_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and fractional values
    x = np.array([3.0, -2.0, 5.3, 1.0, -0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array
    x = np.array(
        [[1.5, -2.5, 3.5], [-4.5, 5.5, -6.5], [0.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D integer array (int32)
    x = np.array([-5, -1, 0, 1, 2, 4, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D int64 array
    x = np.arange(-12, 12, dtype=np.int64).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 0D scalar array
    x = np.array(-4.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D complex64 array
    x = np.array([1 - 3j, -1j, 2 + 0j, -2 - 2j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex128 array
    x = np.array(
        [[1.0 + 2.0j, -3.0 - 4.0j], [0.5 - 0.5j, 2.5 + 1.5j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D float32 array
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Boolean array
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Float16 array
    x = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Array containing zeros, inf, and nan
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: Empty array
    x = np.empty((0, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.square_1"] = square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.square_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.square_1'.")


check_valid('jax.numpy.square', generated_inputs['jax.numpy.square_1'], lib="jax", suffix=1)
