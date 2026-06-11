
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_reduce_sum_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, sum over axis 0
    operand = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 2: 2D float32 array, sum over axis 0
    operand = np.random.randn(3, 4).astype(np.float32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 3: 2D float32 array, sum over axis 1
    operand = np.random.randn(5, 2).astype(np.float32)
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 4: 2D float32 array, sum over axes [0, 1]
    operand = np.random.randn(4, 4).astype(np.float32)
    axes = [0, 1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 5: 3D float32 array, sum over axes [0, 2]
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    axes = [0, 2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 6: 1D float64 array, sum over axis 0
    operand = np.random.randn(10).astype(np.float64)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 7: 2D int16 array with negative values, sum over axis 1
    operand = np.random.randint(-100, 100, size=(4, 4)).astype(np.int16)
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 8: 3D uint8 array, sum over axis 0
    operand = np.random.randint(0, 10, size=(2, 2, 2)).astype(np.uint8)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 9: 4D float32 array, sum over axis 3
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = [3]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 10: 1D float32 array, sum over empty list of axes
    operand = np.array([1.5, 2.5, -3.5], dtype=np.float32)
    axes = []
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    return list_of_inputs

generated_inputs["jax.lax.reduce_sum_1"] = jax_lax_reduce_sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_sum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_sum_1'.")


check_valid('jax.lax.reduce_sum', generated_inputs['jax.lax.reduce_sum_1'], lib="jax", suffix=1)
