
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arccosh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 values >= 1.0
    x = np.array([1.0, 1.5, 2.0, 3.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array of float64 values
    x = np.array([1.0, 2.5, 5.0, 20.0, 100.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array of float32 values
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array of int32 values including values < 1 (returns NaN per documentation)
    x = np.array([[1, 3, -4], [-5, 2, 7]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array of complex64 values
    x = np.array(
        [1.0 + 2.0j, -5.0 + 0.0j, 0.0 + 1.0j, 2.0 - 3.0j], dtype=np.complex64
    )
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D array of complex128 values
    x = np.array(
        [[1.0 + 0.0j, 2.0 + 3.0j], [-1.0 - 1.0j, 0.5 + 0.5j]],
        dtype=np.complex128,
    )
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 3D array of float32 values
    x = np.linspace(1.0, 10.0, 24, dtype=np.float32).reshape(2, 3, 4)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array with values near boundary 1.0 and infinity
    x = np.array([1.0, 1.000001, 10.0, np.inf], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D single-element array (avoids 0-d array)
    x = np.array([2.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array of real values < 1.0
    x = np.array([-10.0, -1.0, 0.0, 0.5, 0.99], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 4D array of float32 values
    x = np.linspace(1.0, 5.0, 16, dtype=np.float32).reshape(2, 2, 2, 2)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 12: 1D array of int64 values >= 1
    x = np.array([1, 2, 5, 10, 50, 100], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.numpy.arccosh_4"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_4'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_4'], lib="jax", suffix=4)
