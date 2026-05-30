
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def flip_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    m = np.random.randn(10).astype(np.float32)
    axis = 0
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    # Input 2: 2D int32 array, axis 1
    m = np.random.randint(-100, 100, size=(5, 5)).astype(np.int32)
    axis = 1
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    # Input 3: 2D float64 array, axis 0
    m = np.random.randn(4, 8).astype(np.float64)
    axis = 0
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    # Input 4: 3D int64 array, axis 2
    m = np.random.randint(-1000, 1000, size=(3, 3, 3)).astype(np.int64)
    axis = 2
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    # Input 5: 3D float32 array, axis -1 (negative axis)
    m = np.random.randn(2, 3, 4).astype(np.float32)
    axis = -1
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    # Input 6: 4D int32 array, axis -2 (negative axis)
    m = np.random.randint(-50, 50, size=(2, 2, 3, 3)).astype(np.int32)
    axis = -2
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    # Input 7: 1D int64 array, axis -1
    m = np.random.randint(-1000, 1000, size=(100,)).astype(np.int64)
    axis = -1
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    # Input 8: 5D float32 array, axis 3
    m = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axis = 3
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    # Input 9: 2D int32 array, axis -2
    m = np.random.randint(-3000, 3000, size=(10, 10)).astype(np.int32)
    axis = -2
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    # Input 10: 3D float64 array, axis 1
    m = np.random.randn(3, 5, 7).astype(np.float64)
    axis = 1
    list_of_inputs.append({"m": copy.deepcopy(m), "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.flip_1"] = flip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flip_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flip_1'.")


check_valid('jax.numpy.flip', generated_inputs['jax.numpy.flip_1'], lib="jax", suffix=1)
