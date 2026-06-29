
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nextafter_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D arrays of the same shape
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: float64, 2D arrays of the same shape
    x1 = np.array([[1.0, -1.0], [0.0, 2.0]], dtype=np.float64)
    x2 = np.array([[2.0, -2.0], [1.0, 1.0]], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: float16, 1D arrays with zeroes
    x1 = np.array([0.0, -0.0, 10.0], dtype=np.float16)
    x2 = np.array([1.0, -1.0, 9.0], dtype=np.float16)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: float32, broadcasting (1, 3) and (3, 3)
    x1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    x2 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: float32, broadcasting (3, 1) and (1, 3)
    x1 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    x2 = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: float64, random 3D arrays
    x1 = np.random.randn(2, 3, 4).astype(np.float64)
    x2 = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: float32, random 4D arrays
    x1 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    x2 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: float32, 0D arrays (representing scalar-like arrays)
    x1 = np.array(0.0, dtype=np.float32)
    x2 = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: float32, arrays with infinity and nan
    x1 = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    x2 = np.array([0.0, 0.0, 1.0, np.inf], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: float64, arrays with very large numbers
    x1 = np.array([1e300, -1e300], dtype=np.float64)
    x2 = np.array([1e308, -1e308], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 11: float32, negative elements moving away from zero
    x1 = np.array([-5.0, -10.0, -100.0], dtype=np.float32)
    x2 = np.array([-5.1, -10.1, -100.1], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.lax.nextafter_1"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.nextafter_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.nextafter_1'.")


check_valid('jax.lax.nextafter', generated_inputs['jax.lax.nextafter_1'], lib="jax", suffix=1)
