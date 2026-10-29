
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reducing over axis 0
    operand = np.array([True, False, True, True, False], dtype=np.bool_)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 2: 2D int32 array with negative values, reducing over axis 1
    operand = np.array([[-10, 20, -30], [5, -15, 25]], dtype=np.int32)
    axes = (1,)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 3: 2D int32 array, reducing over multiple axes (0, 1)
    operand = np.array([[255, 0], [128, 64]], dtype=np.int32)
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 4: 3D int64 array, reducing over axis 0
    operand = np.random.randint(-1000, 1000, size=(2, 3, 4), dtype=np.int64)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 5: 3D int32 array, reducing over non-contiguous axes (0, 2)
    operand = np.random.randint(-100, 100, size=(3, 4, 5), dtype=np.int32)
    axes = (0, 2)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 6: 4D int32 array, reducing over empty axes tuple
    operand = np.random.randint(0, 100, size=(2, 2, 2, 2), dtype=np.int32)
    axes = ()
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 7: 4D int32 array, reducing over axes (1, 2, 3)
    operand = np.random.randint(-128, 127, size=(2, 3, 2, 4), dtype=np.int32)
    axes = (1, 2, 3)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 8: 1D int64 array, reducing over axis 0
    operand = np.array([1, 2, 4, 8, 16, 32], dtype=np.int64)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 9: 5D boolean array, reducing over axes (0, 2, 4)
    operand = np.random.choice([True, False], size=(2, 2, 3, 2, 2)).astype(
        np.bool_
    )
    axes = (0, 2, 4)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 10: 2D int32 array, reducing over axis 0
    operand = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 11: 3D int64 array, reducing over all axes (0, 1, 2)
    operand = np.random.randint(0, 10000, size=(2, 3, 2), dtype=np.int64)
    axes = (0, 1, 2)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

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
