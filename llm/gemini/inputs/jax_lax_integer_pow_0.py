
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def integer_pow_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive power
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float32 array, cube
    x = np.random.randn(3, 4).astype(np.float32)
    y = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar float32 array, power of 0
    x = np.array(5.5, dtype=np.float32)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int32 array, positive power
    x = np.random.randint(1, 10, size=(2, 3, 4)).astype(np.int32)
    y = 4
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 2D float64 array, negative power
    x = np.random.uniform(1.0, 5.0, size=(4, 4)).astype(np.float64)
    y = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D float32 array, power of 5
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = 5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D int64 array, identity power
    x = np.random.randint(-10, 10, size=(5, 5)).astype(np.int64)
    y = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D float32 array with negative values, odd power
    x = np.array([-2.5, -1.0, 0.5, 3.2], dtype=np.float32)
    y = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 3D float32 array, negative power
    x = np.random.uniform(0.5, 2.0, size=(2, 3, 2)).astype(np.float32)
    y = -2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 1D int32 array, power of 2
    x = np.array([2, -3, 4, -5], dtype=np.int32)
    y = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 2D float64 array, power of 4
    x = np.random.randn(3, 3).astype(np.float64)
    y = 4
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.integer_pow"] = integer_pow_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.integer_pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.integer_pow'.")


check_valid('jax.lax.integer_pow', generated_inputs['jax.lax.integer_pow'], lib="jax", suffix=0)
