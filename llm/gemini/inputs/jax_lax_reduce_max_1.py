
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reduce_max_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, reduction over axis [0]
    operand = np.array([-1.0, 2.5, 0.0, 4.2, -3.1], dtype=np.float32)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float64, reduction over axis [0]
    operand = np.random.randn(3, 4).astype(np.float64)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, int32, reduction over axis [1]
    operand = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    axes = [1]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, int64, reduction over multiple axes [0, 1]
    operand = np.random.randint(-100, 100, size=(10, 20)).astype(np.int64)
    axes = [0, 1]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, float32, reduction over axis [2]
    operand = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    axes = [2]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, float64, reduction over axes [0, 2]
    operand = np.random.randn(4, 3, 2).astype(np.float64)
    axes = [0, 2]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, int32, reduction over axis [3]
    operand = np.random.randint(0, 50, size=(2, 2, 3, 3)).astype(np.int32)
    axes = [3]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, float32, empty reduction list (returns copy of operand)
    operand = np.array([1.5, -2.3, 0.1], dtype=np.float32)
    axes = []
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, int64, reduction over all axes [0, 1, 2]
    operand = np.random.randint(-5, 5, size=(3, 3, 3)).astype(np.int64)
    axes = [0, 1, 2]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, float32, reduction over non-consecutive axes [1, 3]
    operand = np.random.randn(2, 3, 2, 4, 2).astype(np.float32)
    axes = [1, 3]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reduce_max_1"] = reduce_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_max_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_max_1'.")


check_valid('jax.lax.reduce_max', generated_inputs['jax.lax.reduce_max_1'], lib="jax", suffix=1)
