
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def exp2_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with standard positive/negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array
    x = np.array([[-1.5, 0.5], [2.5, -3.5]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D int32 array (should be promoted to inexact dtype)
    x = np.array([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D scalar array (float32)
    x = np.array(5.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D float32 array with random small values
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D int64 array
    x = np.array([0, 1, 2, 3, 4, 5, 10], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D float32 array containing zeros and ones
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float32 array with fractional/decimal values
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 3D float64 array with purely negative values
    x = np.array([[[-1.1, -2.2], [-3.3, -4.4]], [[-5.5, -6.6], [-7.7, -8.8]]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float32 array with values leading to underflow (very small result)
    x = np.array([-10.0, -20.0, -50.0, -100.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.exp2_1"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.exp2_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.exp2_1'.")


check_valid('jax.numpy.exp2', generated_inputs['jax.numpy.exp2_1'], lib="jax", suffix=1)
