
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tan_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (float32)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with mixed values (positive, negative, zero)
    x = np.array([0.0, -np.pi / 4, np.pi / 4, -np.pi / 3, np.pi / 6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array (float32)
    x = np.random.uniform(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, size=(3, 4)).astype(
        np.float32
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array (float64)
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array (float16)
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D integer array (int32)
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D integer array (int64)
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex64 array
    x = np.array([1.0 + 2.0j, -0.5 + 0.5j, 0.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Complex128 2D array
    x = np.array([[1.0 - 1.0j, 2.0 + 3.0j], [-2.0 + 0.0j, 0.5 - 0.5j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array containing inf and nan
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty array
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: Boolean array
    x = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.tan_1"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tan_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tan_1'.")


check_valid('jax.numpy.tan', generated_inputs['jax.numpy.tan_1'], lib="jax", suffix=1)
