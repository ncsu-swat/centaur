
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def greater_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array
    x = 0.0
    y = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2: 2D float64 array
    x = 1.5
    y = np.array([[1.0, 2.0], [0.5, 3.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3: Negative float x, 3D float32 array
    x = -3.14
    y = np.random.uniform(-5.0, 5.0, size=(2, 2, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4: Large positive float x, 1D int32 array
    x = 10.0
    y = np.array([5, 10, 15, 20], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5: Float x, 4D float32 array
    x = 0.5
    y = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6: Zero float x, 2D int64 array
    x = 0.0
    y = np.array([[-1, 2], [3, -4]], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7: Float x, 5D float64 array
    x = -1.23
    y = np.random.uniform(-2.0, 2.0, size=(1, 2, 1, 3, 2)).astype(np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8: Float x close to values, 1D float32 array
    x = 2.718
    y = np.array([2.7, 2.71, 2.718, 2.72, 2.8], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9: Large float x, 3D int16 array
    x = 100.0
    y = np.random.randint(0, 150, size=(2, 3, 4)).astype(np.int16)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10: Float x, single element 1D array
    x = -50.5
    y = np.array([-50.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.greater_4"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_4'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_4'], lib="jax", suffix=4)
