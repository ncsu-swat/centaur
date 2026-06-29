
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unstack_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis 0, int32
    x = np.array([10, 20, 30], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 2: 2D array, axis 0, float32
    x = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 3: 2D array, axis 1, float64
    x = np.random.randn(2, 5).astype(np.float64)
    axis = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 4: 3D array, axis 0, int16
    x = np.random.randint(-10, 10, size=(2, 3, 2)).astype(np.int16)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 5: 3D array, axis 2, float32
    x = np.random.randn(2, 2, 4).astype(np.float32)
    axis = 2
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 6: 4D array, negative axis -1, int8
    x = np.random.randint(0, 5, size=(2, 2, 3, 3)).astype(np.int8)
    axis = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 7: 4D array, negative axis -2, uint8
    x = np.random.randint(0, 10, size=(1, 3, 2, 2)).astype(np.uint8)
    axis = -2
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 8: 2D array, axis -1, complex64
    x = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    axis = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 9: 5D array, axis 3, float32
    x = np.random.randn(2, 1, 2, 3, 2).astype(np.float32)
    axis = 3
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 10: 1D array of size 1, axis 0, bool
    x = np.array([True, False, True], dtype=np.bool_)
    axis = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    # Input 11: 3D array, axis 1, float32
    x = np.random.randn(4, 3, 2).astype(np.float32)
    axis = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "axis": axis})

    return list_of_inputs

generated_inputs["jax.lax.unstack"] = unstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.unstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.unstack'.")


check_valid('jax.lax.unstack', generated_inputs['jax.lax.unstack'], lib="jax", suffix=0)
