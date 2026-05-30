
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def size_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis with 1 element
    a = np.random.randn(10).astype(np.float32)
    axis = (0,)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 2: 2D array, int32, axis with 1 element
    a = np.random.randint(-10, 10, size=(5, 6)).astype(np.int32)
    axis = (1,)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 3: 2D array, float64, axis with 2 elements
    a = np.random.randn(4, 4).astype(np.float64)
    axis = (0, 1)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 4: 3D array, bool, axis with 2 elements
    a = np.random.choice([True, False], size=(3, 3, 3))
    axis = (0, 2)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 5: 4D array, float32, axis with negative index
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = (-1,)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 6: 3D array, int64, multiple negative axes
    a = np.random.randint(0, 100, size=(2, 4, 6)).astype(np.int64)
    axis = (-3, -2)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 7: 5D array, complex64, multiple axes
    a = (np.random.randn(2, 2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2, 2)).astype(np.complex64)
    axis = (1, 3, 4)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 8: 1D array, float16, negative axis
    a = np.random.randn(15).astype(np.float16)
    axis = (-1,)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 9: 2D array, uint8, axis is (0,)
    a = np.random.randint(0, 255, size=(10, 20)).astype(np.uint8)
    axis = (0,)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 10: 4D array, int16, all axes specified in tuple
    a = np.random.randint(-5, 5, size=(2, 3, 2, 3)).astype(np.int16)
    axis = (0, 1, 2, 3)
    list_of_inputs.append({"a": a, "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.size_2"] = size_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.size_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.size_2'.")


check_valid('jax.numpy.size', generated_inputs['jax.numpy.size_2'], lib="jax", suffix=2)
