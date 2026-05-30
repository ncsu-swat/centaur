
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, standard x2
    x1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x2 = 0.5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 2: 2D float32 array, x2 is 0.0
    x1 = np.array([[-1.5, 0.0], [3.5, -0.5]], dtype=np.float32)
    x2 = 0.0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 3: 3D float64 array, x2 is 1.0
    x1 = np.array([[[-1.0, 0.0], [1.0, -2.0]], [[0.0, 2.0], [-3.0, 0.0]]], dtype=np.float64)
    x2 = 1.0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 4: 1D int32 array (as "tensor"), x2 is a negative float
    x1 = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    x2 = -0.5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 5: 0D array (scalar-like tensor), x2 is 0.25
    x1 = np.array(0.0, dtype=np.float32)
    x2 = 0.25
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 6: Large 2D float32 array with random values
    x1 = np.random.uniform(-10, 10, size=(50, 50)).astype(np.float32)
    # Inject some zeros
    x1[x1 < -8] = 0.0
    x2 = 0.75
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 7: 4D float32 array, x2 is 2.5
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x2 = 2.5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 8: 1D array of all zeros, x2 is 0.123
    x1 = np.zeros((10,), dtype=np.float32)
    x2 = 0.123
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 9: 1D array of all positive values, x2 is -1.5
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x2 = -1.5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 10: 1D array of all negative values, x2 is 0.5
    x1 = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    x2 = 0.5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 11: 2D int64 array, x2 is 0.99
    x1 = np.array([[-1, 0, 1], [0, -1, 1]], dtype=np.int64)
    x2 = 0.99
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_2"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_2'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_2'], lib="jax", suffix=2)
