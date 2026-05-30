
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def flip_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis [0]
    m = np.arange(10).astype(np.float32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 2: 2D float32 array, axis [1]
    m = np.random.randn(3, 4).astype(np.float32)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 3: 2D int32 array, axis [0, 1]
    m = np.arange(12).reshape(3, 4).astype(np.int32)
    axis = [0, 1]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 4: 3D float64 array, axis [2]
    m = np.random.randn(2, 3, 4).astype(np.float64)
    axis = [2]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 5: 3D boolean array, axis [0, 2]
    m = np.random.choice([True, False], size=(2, 2, 2))
    axis = [0, 2]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 6: 4D complex64 array, axis [-1]
    m = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex64)
    axis = [-1]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 7: 2D float32 array, negative axes [-2, -1]
    m = np.random.randn(5, 5).astype(np.float32)
    axis = [-2, -1]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 8: 1D int16 array, axis [-1]
    m = np.array([1, -2, 3, -4, 5], dtype=np.int16)
    axis = [-1]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 9: 3D uint8 array, axis [1, 2]
    m = np.random.randint(0, 255, size=(2, 4, 4), dtype=np.uint8)
    axis = [1, 2]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 10: 5D float32 array, axis [0, 2, 4]
    m = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axis = [0, 2, 4]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    return list_of_inputs

generated_inputs["jax.numpy.flip_2"] = flip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flip_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flip_2'.")


check_valid('jax.numpy.flip', generated_inputs['jax.numpy.flip_2'], lib="jax", suffix=2)
