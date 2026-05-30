
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_add_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with positive float
    x = 5.0
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2: 2D array with negative float
    x = -2.5
    y = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3: 3D array of float64 with zero float
    x = 0.0
    y = np.ones((2, 2, 2), dtype=np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4: Large positive float and int32 array
    x = 1000.0
    y = np.arange(5, dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5: Very small float and high-dimensional array
    x = 1e-5
    y = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6: Negative float and 2D array with negative values
    x = -10.5
    y = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7: Float and 1D array of float64
    x = 3.14159
    y = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8: Integer-like float and bool array (treated as 0/1)
    x = 1.0
    y = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9: Negative float and 1D single element array
    x = -0.5
    y = np.array([10.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10: Float and 3D array of zeros
    x = 42.42
    y = np.zeros((3, 1, 4), dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.add_4"] = jax_numpy_add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_4'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_4'], lib="jax", suffix=4)
