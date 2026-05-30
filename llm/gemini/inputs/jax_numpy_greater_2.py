
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def greater_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array and a positive float
    x = np.array([1.5, -2.3, 0.0, 4.7], dtype=np.float32)
    y = 2.0
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D float32 array and a negative float
    x = np.random.randn(3, 3).astype(np.float32)
    y = -1.5
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D float64 array and 0.0 float
    x = np.random.randn(2, 2, 2).astype(np.float64)
    y = 0.0
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D int32 array and a float
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    y = 3.5
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 2D int64 array and a float
    x = np.array([[10, 20], [30, 40]], dtype=np.int64)
    y = 25.0
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 4D float32 array and a float
    x = np.random.uniform(-10, 10, size=(2, 2, 3, 3)).astype(np.float32)
    y = 5.25
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 0D scalar array and a float
    x = np.array(4.5, dtype=np.float32)
    y = 4.0
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 1D float16 array and a float
    x = np.array([-5.0, -4.0, -3.0], dtype=np.float16)
    y = -3.5
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Large 1D array of float32 and a float
    x = np.linspace(-100, 100, 1000, dtype=np.float32)
    y = 50.0
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 2D array of zeros and a non-zero float
    x = np.zeros((5, 5), dtype=np.float32)
    y = -0.1
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: 3D array of ints and a very large float
    x = np.ones((2, 3, 4), dtype=np.int32) * 10
    y = 100.0
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.greater_2"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_2'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_2'], lib="jax", suffix=2)
