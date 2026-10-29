
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cbrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D integer array (int32)
    x = np.array([[216, 125, 64], [-27, -8, -1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array (float64)
    x = np.array(-64.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D float64 array with random values
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D float32 array
    x = np.random.uniform(-100, 100, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D float16 array
    x = np.array([0.125, 0.0, -0.125, 1000.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D unsigned integer array (uint8)
    x = np.array([[0, 1, 8], [27, 64, 125]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array with special float values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D int64 array with large integers
    x = np.array(
        [[1000000000, -1000000000], [8000000, -8000000]], dtype=np.int64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D boolean array
    x = np.array([True, False, True], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.cbrt_1"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cbrt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cbrt_1'.")


check_valid('jax.numpy.cbrt', generated_inputs['jax.numpy.cbrt_1'], lib="jax", suffix=1)
