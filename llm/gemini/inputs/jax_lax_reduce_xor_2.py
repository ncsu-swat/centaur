
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reduce_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reduce over axis (0,)
    operand = np.array([True, False, True, False, True], dtype=bool)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with negative values, reduce over axis (0,)
    operand = np.array([-1, 2, -3, 4], dtype=np.int32)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array, reduce over axis (0,)
    operand = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array with negative values, reduce over axis (1,)
    operand = np.array([[-10, 20], [-30, 40]], dtype=np.int64)
    axes = (1,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, reduce over axis (0, 2)
    operand = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int64)
    axes = (0, 2)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D boolean array, reduce over empty axis ()
    operand = np.random.choice([True, False], size=(2, 2, 2, 2))
    axes = ()
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int32 array, reduce over axis (1,)
    operand = np.random.randint(0, 1000, size=(2, 4, 3)).astype(np.int32)
    axes = (1,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32 array, reduce over all axes (0, 1)
    operand = np.random.randint(-128, 127, size=(5, 5)).astype(np.int32)
    axes = (0, 1)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D int32 array, reduce over axis (2, 3, 4)
    operand = np.random.randint(-1000, 1000, size=(2, 2, 3, 3, 3)).astype(np.int32)
    axes = (2, 3, 4)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int64 array, reduce over axis (0,)
    operand = np.array([2**30, -2**30, 2**15], dtype=np.int64)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reduce_xor_2"] = reduce_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_xor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_xor_2'.")


check_valid('jax.lax.reduce_xor', generated_inputs['jax.lax.reduce_xor_2'], lib="jax", suffix=2)
