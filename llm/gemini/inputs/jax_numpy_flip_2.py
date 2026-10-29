
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def flip_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, flip axis 0
    m = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 2: 2D float32 array, flip axis 0
    m = np.random.randn(3, 4).astype(np.float32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 3: 2D float32 array, flip axis 1
    m = np.random.randn(4, 5).astype(np.float32)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 4: 2D float64 array, flip both axes [0, 1]
    m = np.random.randn(5, 5).astype(np.float64)
    axis = [0, 1]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 5: 3D int64 array with negative numbers, flip negative axis [-1]
    m = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int64)
    axis = [-1]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 6: 3D boolean array, flip axes [1, 2]
    m = np.random.choice([True, False], size=(3, 3, 3))
    axis = [1, 2]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 7: 3D complex64 array, flip axes [0, 2]
    m = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(
        np.complex64
    )
    axis = [0, 2]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 8: 4D float32 array, flip negative axes [-3, -1]
    m = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = [-3, -1]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 9: 4D uint8 array, flip all axes [0, 1, 2, 3]
    m = np.random.randint(0, 256, size=(2, 2, 3, 3), dtype=np.uint8)
    axis = [0, 1, 2, 3]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 10: 5D float64 array, flip subset of axes [1, 3]
    m = np.random.randn(1, 2, 3, 2, 1).astype(np.float64)
    axis = [1, 3]
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 11: 1D array, flip negative axis [-1]
    m = np.array([-10.5, 0.0, 15.2, -3.14], dtype=np.float32)
    axis = [-1]
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
