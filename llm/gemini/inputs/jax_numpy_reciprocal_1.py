
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reciprocal_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats
    x = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with negative floats
    x = np.array([-0.5, -1.0, -2.5, -10.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of integers
    x = np.array([[1, 2, 4], [5, 10, 20]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D array (scalar tensor)
    x = np.array(5.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array with floats
    x = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array with negative and positive values
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Complex numbers
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, 0.5 - 0.5j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex128 multi-dimensional
    x = np.array(
        [[1.0j, 2.0 - 1.0j], [3.0 + 0.0j, -4.0j]], dtype=np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Array containing zero and infinity
    x = np.array([0.0, -0.0, np.inf, -np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Int64 1D array
    x = np.array([-100, -10, -1, 1, 10, 100], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Float16 array
    x = np.array([[0.25, 0.5], [1.5, 2.5]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.reciprocal_1"] = reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reciprocal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reciprocal_1'.")


check_valid('jax.numpy.reciprocal', generated_inputs['jax.numpy.reciprocal_1'], lib="jax", suffix=1)
