
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reduce_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reduce over axis [0]
    operand = np.array([True, False, True, False], dtype=bool)
    axes = [0]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 2: 2D int32 array, reduce over axis [0]
    operand = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axes = [0]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 3: 3D int8 array, reduce over axes [0, 2]
    operand = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    axes = [0, 2]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 4: 2D int16 array, reduce over axis [1]
    operand = np.array([[12, 34], [56, 78]], dtype=np.int16)
    axes = [1]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 5: 3D boolean array, reduce over axes [1, 2]
    operand = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=bool)
    axes = [1, 2]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 6: 1D int16 array, empty axes
    operand = np.array([10, 20, 30], dtype=np.int16)
    axes = []
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 7: 4D int64 array, reduce over axis [2]
    operand = np.ones((2, 2, 2, 2), dtype=np.int64)
    axes = [2]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 8: 2D int32 array, reduce over axes [0, 1]
    operand = np.array([[100, 200], [300, 400]], dtype=np.int32)
    axes = [0, 1]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 9: 1D int32 array with negative values, reduce over axis [0]
    operand = np.array([-1, -2, -3, 4], dtype=np.int32)
    axes = [0]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 10: 2D int8 array with negative values, reduce over axis [1]
    operand = np.array([[-5, 12], [-120, -1]], dtype=np.int8)
    axes = [1]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    return list_of_inputs

generated_inputs["jax.lax.reduce_xor_1"] = reduce_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_xor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_xor_1'.")


check_valid('jax.lax.reduce_xor', generated_inputs['jax.lax.reduce_xor_1'], lib="jax", suffix=1)
