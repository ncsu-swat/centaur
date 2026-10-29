
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def add_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive float scalar
    x = 2.5
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 array with negative float scalar
    x = -4.0
    y = np.array([[-1, 0, 1], [2, -3, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D float64 array with zero float scalar
    x = 0.0
    y = np.ones((2, 3, 4), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar array with positive float scalar
    x = 10.5
    y = np.array(5.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 4D float32 array with negative float scalar
    x = -1.23
    y = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 1D empty array with float scalar
    x = 3.14159
    y = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D array containing infinity and NaN with small float scalar
    x = 1e-5
    y = np.array([[np.inf, -np.inf], [np.nan, 0.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 2D uint8 array with large float scalar
    x = 255.0
    y = np.arange(6, dtype=np.uint8).reshape((2, 3))
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 5D float32 array with float scalar
    x = 0.5
    y = np.zeros((1, 2, 1, 3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D complex64 array with float scalar
    x = 1.5
    y = np.array(
        [[1.0 + 2.0j, -3.0 + 0.0j], [0.0 - 1.0j, 4.0 + 4.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 1D bool array with float scalar
    x = 2.0
    y = np.array([True, False, True], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.add_4"] = add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_4'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_4'], lib="jax", suffix=4)
