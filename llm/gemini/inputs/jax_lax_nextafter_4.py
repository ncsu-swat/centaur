
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nextafter_inputs():
    list_of_inputs = []

    # Input 1: Basic positive float and 1D tensor (float64)
    x1 = 1.0
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: Zero and 2D tensor (float64)
    x1 = 0.0
    x2 = np.array([[-1.0, 1.0], [2.0, -2.0]], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: Negative float and 3D tensor (float64)
    x1 = -5.5
    x2 = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: Large positive float and 1D tensor (float64)
    x1 = 1e10
    x2 = np.array([1e11, 1e9, 0.0], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Small float and 4D tensor (float64)
    x1 = 1e-5
    x2 = np.ones((2, 2, 2, 2), dtype=np.float64) * 1e-4
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Float and float32 tensor
    x1 = 2.5
    x2 = np.array([1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Negative zero and 1D tensor
    x1 = -0.0
    x2 = np.array([-1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: Infinity direction
    x1 = 1.5
    x2 = np.array([np.inf, -np.inf], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Nan direction
    x1 = 0.5
    x2 = np.array([np.nan, 1.0], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: High-dimensional tensor
    x1 = -100.0
    x2 = np.zeros((1, 5, 1, 5), dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.lax.nextafter_4"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.nextafter_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.nextafter_4'.")


check_valid('jax.lax.nextafter', generated_inputs['jax.lax.nextafter_4'], lib="jax", suffix=4)
