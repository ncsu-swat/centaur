
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D array of floats, positive float
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = 2.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 2: 2D array of ints, negative float
    x = np.array([[1, -2], [3, 4]], dtype=np.int32)
    y = -2.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 3: 3D array of floats, float is zero
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = 0.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 4: 1D array of float64, float is positive
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    y = 1.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 5: Scalar-like array (0D array), float is positive
    x = np.array(5.0, dtype=np.float32)
    y = 5.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 6: 4D array, float32, positive float
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    y = 0.5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 7: Array with negative and positive values, float64, negative float
    x = np.arange(-5, 5).astype(np.float64)
    y = -3.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 8: 2D array of float16, positive float
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
    y = 2.5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 9: Large 1D array of float32, float is zero
    x = np.linspace(-10.0, 10.0, 100).astype(np.float32)
    y = 0.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 10: 3D array of int64, float is positive
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = 4.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.not_equal_2"] = jax_numpy_not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_2'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_2'], lib="jax", suffix=2)
