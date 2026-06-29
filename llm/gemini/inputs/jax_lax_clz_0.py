
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def clz_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar, int32
    x = np.array(5, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array, int32, positive values
    x = np.array([0, 1, 2, 4, 8, 16, 1024], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array, int32, mixed positive and negative
    x = np.array([[-1, 0, 1], [-128, 127, 64]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D array, int64
    x = np.array([0, 1, -1, 9223372036854775807, -9223372036854775808], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array, int64, random values
    x = np.random.randint(-1000, 1000, size=(2, 3, 4), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array, int32
    x = np.random.randint(-100, 100, size=(2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array, int64, all zeros
    x = np.zeros((5, 5), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 3D array, int32
    x = np.array([[[100, -200], [300, -400]], [[500, -600], [700, -800]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D array, int32
    x = np.random.randint(-10, 10, size=(1, 2, 1, 3, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array, int64, large size
    x = np.random.randint(-10000, 10000, size=(10, 10), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.clz"] = clz_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.clz' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.clz'.")


check_valid('jax.lax.clz', generated_inputs['jax.lax.clz'], lib="jax", suffix=0)
