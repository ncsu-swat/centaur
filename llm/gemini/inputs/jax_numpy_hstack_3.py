
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hstack_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, dtype=None
    tup = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": None}))

    # Input 2: 2D array, float32, dtype=None
    tup = np.random.randn(3, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": None}))

    # Input 3: 2D array, int32, dtype=float32
    tup = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": np.float32}))

    # Input 4: 3D array, float64, dtype=None
    tup = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": None}))

    # Input 5: 1D array, int32 with negative values, dtype=int32
    tup = np.array([-1, -2, -3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": np.int32}))

    # Input 6: 4D array, float32, dtype=float64
    tup = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": np.float64}))

    # Input 7: 2D array, int32 with negative values, dtype=None
    tup = np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": None}))

    # Input 8: 1D array with single element, int32, dtype=int32
    tup = np.array([42], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": np.int32}))

    # Input 9: 2D array, float16, dtype=None
    tup = np.random.randn(4, 2).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": None}))

    # Input 10: 3D array, float32, dtype=float32
    tup = np.random.randn(3, 2, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": np.float32}))

    return list_of_inputs

generated_inputs["jax.numpy.hstack_3"] = hstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hstack_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hstack_3'.")


check_valid('jax.numpy.hstack', generated_inputs['jax.numpy.hstack_3'], lib="jax", suffix=3)
