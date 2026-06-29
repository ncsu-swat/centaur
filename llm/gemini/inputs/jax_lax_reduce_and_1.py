
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_reduce_and_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reduction over axis 0
    operand = np.array([True, False, True, True], dtype=bool)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean array, reduction over axis 0
    operand = np.array([[True, True], [False, True]], dtype=bool)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean array, reduction over axis 1
    operand = np.array([[True, True], [False, True]], dtype=bool)
    axes = [1]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D boolean array, reduction over axes [0, 1]
    operand = np.array([[True, True], [True, True]], dtype=bool)
    axes = [0, 1]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 array, reduction over axis 1 (positive values)
    operand = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axes = [1]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int32 array, reduction over axes [0, 2] (negative values)
    operand = np.array([[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]], dtype=np.int32)
    axes = [0, 2]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D boolean array, reduction over empty axes []
    operand = np.ones((2, 2, 2, 2), dtype=bool)
    axes = []
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int16 array, reduction over axis 0
    operand = np.array([[12, 15], [7, 9]], dtype=np.int16)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D uint8 array, reduction over axis 0
    operand = np.array([255, 255, 254], dtype=np.uint8)
    axes = [0]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D boolean array, reduction over axes [1, 3]
    operand = np.ones((2, 3, 2, 3, 2), dtype=bool)
    axes = [1, 3]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D int64 array, reduction over axis 2
    operand = np.arange(12, dtype=np.int64).reshape((2, 2, 3))
    axes = [2]
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reduce_and_1"] = generate_reduce_and_inputs()

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
