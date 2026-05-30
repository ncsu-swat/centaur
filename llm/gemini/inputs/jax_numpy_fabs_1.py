
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fabs_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([-1.5, 2.3, -0.0, 4.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array with negative values
    x = np.array([[-3.4, -5.6], [7.8, -9.1]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D int32 array
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D int16 array
    x = np.array([[[-1, 2], [-3, 4]], [[-5, 6], [-7, 8]]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Boolean array
    x = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D scalar-like array
    x = np.array(-42.42, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D float32 array with random values (including negatives)
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Array containing inf and nan values
    x = np.array([-np.inf, np.inf, np.nan, -0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large float64 values
    x = np.array([-1e10, 2.5e15, -3.8e20], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: float16 array
    x = np.array([-0.5, -1.5, 2.5], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.fabs_1"] = fabs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fabs_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fabs_1'.")


check_valid('jax.numpy.fabs', generated_inputs['jax.numpy.fabs_1'], lib="jax", suffix=1)
