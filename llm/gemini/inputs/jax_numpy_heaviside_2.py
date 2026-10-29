
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def heaviside_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x1 = np.array([-2.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float32)
    x2 = 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float64 array
    x1 = np.array([[-1.5, 0.0, 2.5], [0.0, -3.2, 4.1]], dtype=np.float64)
    x2 = 1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 3D int32 array
    x1 = np.array(
        [[[-2, 0], [1, -1]], [[0, 3], [-4, 0]]],
        dtype=np.int32,
    )
    x2 = 0.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 1D int64 array
    x1 = np.array([-10, 0, 10, -5, 0, 5], dtype=np.int64)
    x2 = 0.75
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 4D float32 array
    x1 = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 4)).astype(np.float32)
    x1[0, 0, 0, 0] = 0.0
    x2 = -0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D float32 scalar array
    x1 = np.array(0.0, dtype=np.float32)
    x2 = 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 2D float16 array
    x1 = np.array([[-1.0, 1.0], [0.0, -0.0]], dtype=np.float16)
    x2 = 2.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D array of all zeros
    x1 = np.zeros(10, dtype=np.float32)
    x2 = 0.42
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 3D float64 array with negative and positive values
    x1 = np.linspace(-3.0, 3.0, num=24, dtype=np.float64).reshape((2, 3, 4))
    x2 = -1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 2D int16 array
    x1 = np.array([[-100, 0, 100], [0, -50, 50]], dtype=np.int16)
    x2 = 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 5D float32 array
    x1 = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    x2 = 0.125
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.heaviside_2"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_2'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_2'], lib="jax", suffix=2)
