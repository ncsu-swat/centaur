
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atan_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive float32 values
    x = np.array([0.1, 0.5, 1.0, 2.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array with negative float32 values
    x = np.array([-0.1, -0.5, -1.0, -2.0, -10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array with mixed float64 values
    x = np.array([[-1.5, 0.0, 1.5], [-2.5, 3.0, -0.5]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D array (scalar equivalent)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array of float32
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array of int32
    x = np.array([-5, -1, 0, 1, 5], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array containing special values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array with very small values (near zero)
    x = np.array([1e-5, -1e-5, 1e-9, -1e-9], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D array of float16
    x = np.random.uniform(-2.0, 2.0, size=(2, 2, 2, 2)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array of zeros
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 5D array of float32
    x = np.random.uniform(-10.0, 10.0, size=(1, 2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.atan"] = atan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atan'.")


check_valid('jax.numpy.atan', generated_inputs['jax.numpy.atan'], lib="jax", suffix=0)
