
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cbrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([1.0, 8.0, 27.0, 64.0, 125.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D float64 array with negative values
    x = np.array([-1.0, -8.0, -27.0, -64.0, -125.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of integers
    x = np.array([[216, 125, 64], [-27, -8, -1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array including zero and positive/negative values
    x = np.array([-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D array with single element
    x = np.array([1000.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D array of float64 negative values
    x = np.array([-343.0, -512.0, -729.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D array of float32
    x = np.arange(1, 28, dtype=np.float32).reshape((3, 3, 3))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array with large and small magnitudes
    x = np.array([1e9, -1e9, 1e-12, -1e-12], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array with boolean values
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array with special float values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 4D array of float32
    x = np.linspace(-100, 100, 16, dtype=np.float32).reshape((2, 2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.cbrt_5"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cbrt_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cbrt_5'.")


check_valid('jax.numpy.cbrt', generated_inputs['jax.numpy.cbrt_5'], lib="jax", suffix=5)
