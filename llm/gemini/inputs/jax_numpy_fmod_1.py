
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x1 = np.array([-4.75, -2.5, 0.0, 2.5, 4.75], dtype=np.float32)
    x2 = np.array([3.0, 2.0, 1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 2D float64 arrays with negative divisors
    x1 = np.array([[5.5, -3.2], [7.1, -8.9]], dtype=np.float64)
    x2 = np.array([[-2.0, -1.5], [3.0, -2.5]], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 1D int32 arrays with broadcasting (scalar-like array for x2)
    x1 = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    x2 = np.array([3], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 2D arrays with broadcasting along axis 0
    x1 = np.array([[12, -15, 7], [-8, 22, -11]], dtype=np.int64)
    x2 = np.array([[4, 5, 3]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: 3D float32 arrays
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    x2 = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: 0D scalar arrays
    x1 = np.array(-7.5, dtype=np.float32)
    x2 = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: 4D float64 arrays with broadcast dimensions
    x1 = np.random.randn(2, 1, 4, 3).astype(np.float64)
    x2 = np.random.uniform(0.5, 2.0, size=(1, 3, 1, 3)).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: 1D float32 arrays with large values
    x1 = np.array([1000.5, -2500.25, 3333.33, -4000.0], dtype=np.float32)
    x2 = np.array([30.0, 12.0, -15.0, 25.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Mixed signs with int32 dtype
    x1 = np.array([-300, -150, 0, 150, 300], dtype=np.int32)
    x2 = np.array([70, -70, 70, -70, 70], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: 2D column and row broadcast float32
    x1 = np.arange(1, 6, dtype=np.float32).reshape(5, 1)
    x2 = np.arange(2, 6, dtype=np.float32).reshape(1, 4)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 11: 3D int64 arrays with broadcast
    x1 = np.array([[[10, 20], [30, 40]]], dtype=np.int64)
    x2 = np.array([[[3]], [[7]]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs


generated_inputs["jax.numpy.fmod_1"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_1'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_1'], lib="jax", suffix=1)
