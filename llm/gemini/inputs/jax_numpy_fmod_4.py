
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x1 = 5.5
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 2: 2D float64 array with negative numbers
    x1 = -7.2
    x2 = np.array([[1.5, -2.0], [3.1, -4.0]], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 3: Zero dividend, 1D array
    x1 = 0.0
    x2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 4: Positive dividend with mixed sign divisor
    x1 = 10.0
    x2 = np.array([-3.0, 3.0, -4.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 5: 3D float32 array
    x1 = -15.5
    x2 = np.full((2, 3, 4), 4.0, dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 6: 1D array with linspace values
    x1 = 3.141592653589793
    x2 = np.linspace(0.5, 5.0, 10, dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 7: 4D float32 array
    x1 = 42.0
    x2 = np.array([[[[2.0, 3.0], [4.0, 5.0]]]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 8: float16 precision 2D array
    x1 = 1.25
    x2 = np.array([[2.5, 0.5, 1.25], [5.0, 0.25, 0.1]], dtype=np.float16)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 9: Single-element 1D array
    x1 = -100.5
    x2 = np.array([10.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 10: 3D float64 array with both positive and negative values
    x1 = 2.718281828459045
    x2 = np.array([[[-1.5, 2.5]], [[-3.5, 4.5]]], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 11: 2D random positive array
    x1 = 1000.0
    x2 = np.random.uniform(1.0, 50.0, size=(5, 5)).astype(np.float32)
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
