
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_and_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reduce over axis 0
    operand = np.array([True, False, True, True], dtype=np.bool_)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 2: 1D int32 array with negative values, reduce over axis 0
    operand = np.array([-1, 7, -3, 15], dtype=np.int32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 3: 2D boolean array, reduce over axis 1
    operand = np.array([[True, True], [False, True], [True, False]], dtype=np.bool_)
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 4: 2D int64 array, reduce over axis 0
    operand = np.array([[-10, 20, 30], [-1, 5, -8]], dtype=np.int64)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 5: 2D int32 array, reduce over multiple axes [0, 1]
    operand = np.array([[255, 127], [63, 31]], dtype=np.int32)
    axes = [0, 1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 6: 3D int32 array with negative values, reduce over axes [1, 2]
    operand = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    axes = [1, 2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 7: 3D boolean array, reduce over empty axes sequence
    operand = np.ones((2, 2, 2), dtype=np.bool_)
    axes = []
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 8: 4D int64 array, reduce over non-adjacent axes [0, 3]
    operand = np.random.randint(-100, 100, size=(2, 3, 2, 4), dtype=np.int64)
    axes = [0, 3]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 9: 0D scalar int32 array, reduce over empty axes
    operand = np.array(-42, dtype=np.int32)
    axes = []
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 10: 5D int32 array, reduce over axis [2]
    operand = np.random.randint(0, 1000, size=(1, 2, 3, 1, 2), dtype=np.int32)
    axes = [2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    return list_of_inputs


generated_inputs["jax.lax.reduce_and_1"] = reduce_and_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_and_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_and_1'.")


check_valid('jax.lax.reduce_and', generated_inputs['jax.lax.reduce_and_1'], lib="jax", suffix=1)
