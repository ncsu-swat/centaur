
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reduce_or_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reduce over axis 0
    operand = np.array([True, False, True, False], dtype=bool)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean array, reduce over axis 1
    operand = np.random.choice([True, False], size=(3, 4)).astype(bool)
    axes = [1]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array, reduce over axis 0
    operand = np.random.randint(0, 255, size=(5, 5)).astype(np.int32)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array with negative values, reduce over axes [0, 2]
    operand = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int32)
    axes = [0, 2]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D int32 array, reduce over axis 0
    operand = np.array([-1, 0, 1, 2], dtype=np.int32)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D boolean array, reduce over multiple axes [1, 2, 3]
    operand = np.random.choice([True, False], size=(2, 3, 2, 4)).astype(bool)
    axes = [1, 2, 3]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int64 array, reduce over empty list (no axes reduced)
    operand = np.random.randint(-1000, 1000, size=(3, 3)).astype(np.int64)
    axes = []
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D boolean array, reduce over axis 1
    operand = np.random.choice([True, False], size=(4, 2, 3)).astype(bool)
    axes = [1]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int64 array, reduce over axis 0
    operand = np.random.randint(-100000, 100000, size=(6, 2)).astype(np.int64)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D boolean array, reduce over axes [0, 2, 4]
    operand = np.random.choice([True, False], size=(2, 2, 2, 2, 2)).astype(bool)
    axes = [0, 2, 4]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D int32 array, reduce over axes [1, 2]
    operand = np.random.randint(-50, 50, size=(4, 4, 4)).astype(np.int32)
    axes = [1, 2]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reduce_or_1"] = reduce_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_or_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_or_1'.")


check_valid('jax.lax.reduce_or', generated_inputs['jax.lax.reduce_or_1'], lib="jax", suffix=1)
