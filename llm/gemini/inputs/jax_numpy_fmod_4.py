
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmod_inputs():
    list_of_inputs = []

    # Input 1: positive float, 1D float32 array (positive elements)
    x1 = 5.0
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 2: negative float, 1D float32 array (positive elements)
    x1 = -5.0
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 3: positive float, 2D float32 array with mixed signs
    x1 = 7.5
    x2 = np.array([[2.0, -3.0], [1.5, -4.5]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 4: negative float, 2D float64 array with mixed signs
    x1 = -8.2
    x2 = np.array([[-2.5, 3.1], [-1.2, 4.0]], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 5: float, 3D float32 array with positive elements
    x1 = 12.0
    x2 = np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 6: zero float, 1D float32 array
    x1 = 0.0
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 7: large float, 2D int32 array (integer tensor)
    x1 = 100.5
    x2 = np.array([[10, 20], [30, 40]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 8: small float, 4D float32 array
    x1 = 0.5
    x2 = np.random.uniform(0.1, 1.0, size=(2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 9: float, 1D float64 array
    x1 = float(3.14159)
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 10: negative float, 3D float64 array
    x1 = -100.0
    x2 = np.random.uniform(-10.0, -1.0, size=(2, 3, 2)).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.fmod_4"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_4'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_4'], lib="jax", suffix=4)
