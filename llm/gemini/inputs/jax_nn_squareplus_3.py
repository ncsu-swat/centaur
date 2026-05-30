
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def squareplus_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array, scalar b
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    b = np.array(4.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b})

    # Input 2: float32, 2D array, scalar b
    x = np.random.randn(3, 4).astype(np.float32)
    b = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b})

    # Input 3: float32, 3D array, scalar b
    x = np.random.randn(2, 3, 3).astype(np.float32)
    b = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b})

    # Input 4: float64, 1D array, 1D array b (broadcasting)
    x = np.random.randn(5).astype(np.float64)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b})

    # Input 5: float64, 2D array, scalar b
    x = np.random.randn(4, 4).astype(np.float64)
    b = np.array(0.1, dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b})

    # Input 6: float16, 3D array, scalar b
    x = np.random.randn(2, 3, 4).astype(np.float16)
    b = np.array(4.0, dtype=np.float16)
    list_of_inputs.append({"x": x, "b": b})

    # Input 7: float32, 1D array with wider range of negative values, scalar b
    x = np.random.uniform(-10.0, 10.0, size=(10,)).astype(np.float32)
    b = np.array(5.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b})

    # Input 8: float32, 4D array, scalar b
    x = np.random.randn(1, 5, 5, 3).astype(np.float32)
    b = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b})

    # Input 9: float32, 2D array, element-wise b of the same shape
    x = np.random.randn(3, 3).astype(np.float32)
    b = np.random.uniform(0.1, 5.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "b": b})

    # Input 10: float32, large 1D array, scalar b
    x = np.random.randn(100).astype(np.float32)
    b = np.array(10.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b})

    return list_of_inputs

generated_inputs["jax.nn.squareplus_3"] = squareplus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.squareplus_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.squareplus_3'.")


check_valid('jax.nn.squareplus', generated_inputs['jax.nn.squareplus_3'], lib="jax", suffix=3)
