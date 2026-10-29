
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nextafter_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, positive direction
    x1 = 0.0
    x2 = np.array([1.0, 2.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D array, float32, mixed directions
    x1 = 1.0
    x2 = np.array([[2.0, 0.5], [-1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 3D array of zeros, float32
    x1 = -1.5
    x2 = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 1D array, float64, infinity values
    x1 = 2.0
    x2 = np.array([-np.inf, np.inf, 0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 1D array of ones, float64
    x1 = 0.0
    x2 = np.ones((5,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 4D array, float32, random values
    x1 = 100.0
    x2 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 1D array with zero, float32, negative zero x1
    x1 = -0.0
    x2 = np.array([0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D array, float64, linearly spaced values
    x1 = 3.141592653589793
    x2 = np.linspace(-10.0, 10.0, 8, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 3D array with singleton dimensions, float32
    x1 = 1e-5
    x2 = np.array([[[1.0]], [[-1.0]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 0D scalar array, float32
    x1 = -50.0
    x2 = np.array(5.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

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
