
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_reduce_sum_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, sum over axis 0
    operand = np.random.randn(10).astype(np.float32)
    axes = (0,)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 2: 2D int32 array with negative values, sum over axis 1
    operand = np.random.randint(-50, 50, size=(5, 8)).astype(np.int32)
    axes = (1,)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 3: 2D float64 array, sum over both axes (0, 1)
    operand = np.random.randn(4, 4).astype(np.float64)
    axes = (0, 1)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 4: 3D float16 array, sum over axis 2
    operand = np.random.randn(2, 3, 4).astype(np.float16)
    axes = (2,)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 5: 3D int16 array with negative values, sum over axes (0, 2)
    operand = np.random.randint(-100, 100, size=(3, 3, 3)).astype(np.int16)
    axes = (0, 2)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 6: 4D float32 array, sum over axes (1, 3)
    operand = np.random.randn(2, 4, 2, 3).astype(np.float32)
    axes = (1, 3)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 7: 2D float32 array, sum over empty axes tuple
    operand = np.random.randn(3, 5).astype(np.float32)
    axes = ()
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 8: 5D int32 array, sum over axis 4
    operand = np.random.randint(0, 10, size=(2, 2, 3, 2, 4)).astype(np.int32)
    axes = (4,)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 9: 3D float64 array, sum over axes (0, 1, 2)
    operand = np.random.randn(2, 2, 2).astype(np.float64)
    axes = (0, 1, 2)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 10: 1D uint8 array, sum over axis 0
    operand = np.random.randint(0, 255, size=(100,)).astype(np.uint8)
    axes = (0,)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    # Input 11: 4D float32 array, sum over axes (0, 2, 3)
    operand = np.random.randn(2, 5, 3, 4).astype(np.float32)
    axes = (0, 2, 3)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": axes})

    return list_of_inputs

generated_inputs["jax.lax.reduce_sum_2"] = generate_reduce_sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_sum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_sum_2'.")


check_valid('jax.lax.reduce_sum', generated_inputs['jax.lax.reduce_sum_2'], lib="jax", suffix=2)
