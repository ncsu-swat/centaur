
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def size_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, axis = 0
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 2: 2D float32 array, axis = 1
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 1
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 3: 2D int64 array, negative axis
    a = np.arange(10).reshape(2, 5).astype(np.int64)
    axis = -1
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 4: 3D float64 array, axis = 2
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = 2
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 5: 3D bool array, negative axis
    a = np.random.choice([True, False], size=(4, 3, 2)).astype(np.bool_)
    axis = -3
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 6: 4D float32 array, axis = 1
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = 1
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 7: 5D float64 array, negative axis
    a = np.random.randn(1, 2, 3, 4, 5).astype(np.float64)
    axis = -1
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 8: 1D int32 array, axis = 0
    a = np.arange(8, dtype=np.int32)
    axis = 0
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 9: 2D int64 array with 0 dimension (empty), axis = 0
    a = np.empty((0, 5), dtype=np.int64)
    axis = 0
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 10: 3D float32 array, axis = 0
    a = np.random.randn(5, 5, 5).astype(np.float32)
    axis = 0
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 11: 2D int32 array, axis = 1
    a = np.random.randint(0, 100, size=(10, 20)).astype(np.int32)
    axis = 1
    list_of_inputs.append({"a": a, "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.size_1"] = size_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.size_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.size_1'.")


check_valid('jax.numpy.size', generated_inputs['jax.numpy.size_1'], lib="jax", suffix=1)
