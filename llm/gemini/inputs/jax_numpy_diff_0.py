
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def diff_inputs():
    list_of_inputs = []

    # Input 1: 1D array, n=1, axis=0, int32
    a = np.random.randint(-10, 10, size=(10,)).astype(np.int32)
    prepend = np.random.randint(-10, 10, size=(2,)).astype(np.int32)
    append = np.random.randint(-10, 10, size=(3,)).astype(np.int32)
    input_dict = {
        "a": a,
        "n": 1,
        "axis": 0,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, n=2, axis=-1, float32
    a = np.random.randn(8).astype(np.float32)
    prepend = np.random.randn(1).astype(np.float32)
    append = np.random.randn(2).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 2,
        "axis": -1,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, n=1, axis=1, float64
    a = np.random.randn(4, 5).astype(np.float64)
    prepend = np.random.randn(4, 1).astype(np.float64)
    append = np.random.randn(4, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 1,
        "axis": 1,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, n=2, axis=0, int64
    a = np.random.randint(-100, 100, size=(6, 4)).astype(np.int64)
    prepend = np.random.randint(-100, 100, size=(2, 4)).astype(np.int64)
    append = np.random.randint(-100, 100, size=(1, 4)).astype(np.int64)
    input_dict = {
        "a": a,
        "n": 2,
        "axis": 0,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, n=1, axis=2, float32
    a = np.random.randn(3, 3, 3).astype(np.float32)
    prepend = np.random.randn(3, 3, 1).astype(np.float32)
    append = np.random.randn(3, 3, 1).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 1,
        "axis": 2,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, n=0 (no difference), axis=1, int32
    a = np.random.randint(0, 10, size=(2, 4, 3)).astype(np.int32)
    prepend = np.random.randint(0, 10, size=(2, 1, 3)).astype(np.int32)
    append = np.random.randint(0, 10, size=(2, 2, 3)).astype(np.int32)
    input_dict = {
        "a": a,
        "n": 0,
        "axis": 1,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, n=1, axis=3, float32
    a = np.random.randn(2, 2, 3, 4).astype(np.float32)
    prepend = np.random.randn(2, 2, 3, 1).astype(np.float32)
    append = np.random.randn(2, 2, 3, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 1,
        "axis": 3,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with large values, n=1, axis=-1, float32
    a = (np.random.randn(5, 5) * 1000).astype(np.float32)
    prepend = (np.random.randn(5, 1) * 1000).astype(np.float32)
    append = (np.random.randn(5, 1) * 1000).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 1,
        "axis": -1,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, n=3, axis=0, int64
    a = np.random.randint(-50, 50, size=(15,)).astype(np.int64)
    prepend = np.random.randint(-50, 50, size=(3,)).astype(np.int64)
    append = np.random.randint(-50, 50, size=(3,)).astype(np.int64)
    input_dict = {
        "a": a,
        "n": 3,
        "axis": 0,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, n=2, axis=-2, float64
    a = np.random.randn(2, 5, 2).astype(np.float64)
    prepend = np.random.randn(2, 1, 2).astype(np.float64)
    append = np.random.randn(2, 1, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 2,
        "axis": -2,
        "prepend": prepend,
        "append": append
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.diff"] = diff_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diff' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diff'.")


check_valid('jax.numpy.diff', generated_inputs['jax.numpy.diff'], lib="jax", suffix=0)
