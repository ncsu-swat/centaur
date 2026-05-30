
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def power_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive integer power
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = 3
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 2: 2D int32 array, positive integer power
    x1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    x2 = 2
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 3: 3D float64 array, negative integer power (valid for float bases)
    x1 = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
    x2 = -2
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 4: 1D float32 array with negative values, positive odd power
    x1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x2 = 3
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 5: 0D float32 array, power 0
    x1 = np.array(5.5, dtype=np.float32)
    x2 = 0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 6: 2D complex64 array, positive power
    x1 = np.array([[1 + 1j, 2 - 1j], [0.5j, -1j]], dtype=np.complex64)
    x2 = 2
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 7: 4D float32 array, large dimensions, positive power
    x1 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    x2 = 4
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 8: 1D int64 array, power 1
    x1 = np.array([10, 20, 30, 40], dtype=np.int64)
    x2 = 1
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 9: 2D float32 array containing zeros, positive power
    x1 = np.array([[0.0, 1.5], [-2.5, 0.0]], dtype=np.float32)
    x2 = 5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 10: 3D int16 array with positive values, positive power
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    x2 = 3
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.power_2"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_2'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_2'], lib="jax", suffix=2)
