
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vstack_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, dtype=None
    tup = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"tup": tup, "dtype": None})

    # Input 2: 1D array, int32, dtype=float32
    tup = np.random.randint(-10, 10, size=(5,)).astype(np.int32)
    list_of_inputs.append({"tup": tup, "dtype": np.float32})

    # Input 3: 3D array, float64, dtype=None
    tup = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"tup": tup, "dtype": None})

    # Input 4: 2D array, int64, dtype=int32
    tup = np.random.randint(-100, 100, size=(5, 2)).astype(np.int64)
    list_of_inputs.append({"tup": tup, "dtype": np.int32})

    # Input 5: 2D array, float32, dtype=None (negative values included)
    tup = np.random.uniform(-5.0, 5.0, size=(1, 10)).astype(np.float32)
    list_of_inputs.append({"tup": tup, "dtype": None})

    # Input 6: 1D array, float32, dtype=float64
    tup = np.random.randn(10).astype(np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.float64})

    # Input 7: 3D array, float32, dtype=None
    tup = np.random.randn(3, 5, 5).astype(np.float32)
    list_of_inputs.append({"tup": tup, "dtype": None})

    # Input 8: 2D array, float32, dtype=float32
    tup = np.random.randn(4, 3).astype(np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.float32})

    # Input 9: 2D array, int32, dtype=None
    tup = np.random.randint(0, 50, size=(10, 2)).astype(np.int32)
    list_of_inputs.append({"tup": tup, "dtype": None})

    # Input 10: 1D array, int8, dtype=int32
    tup = np.random.randint(-10, 10, size=(2,)).astype(np.int8)
    list_of_inputs.append({"tup": tup, "dtype": np.int32})

    return list_of_inputs

generated_inputs["jax.numpy.vstack_1"] = vstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vstack_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vstack_1'.")


check_valid('jax.numpy.vstack', generated_inputs['jax.numpy.vstack_1'], lib="jax", suffix=1)
