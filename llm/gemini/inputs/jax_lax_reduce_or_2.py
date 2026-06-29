
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_reduce_or_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reduce over axis 0
    operand = np.array([True, False, True, False], dtype=np.bool_)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean array, reduce over axis 1
    operand = np.array([[True, False], [False, False], [True, True]], dtype=np.bool_)
    axes = (1,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array with positive integers, reduce over axis 0
    operand = np.array([[1, 2, 4], [8, 16, 32]], dtype=np.int32)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int64 array with mixed values, reduce over axes (1, 2)
    operand = np.arange(-5, 7, dtype=np.int64).reshape((2, 2, 3))
    axes = (1, 2)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D int32 array with negative values, reduce over axis 0
    operand = np.array([-1, -2, 3, 4], dtype=np.int32)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D boolean array, reduce over axes (0, 2, 3)
    operand = np.random.choice([True, False], size=(2, 3, 2, 2)).astype(np.bool_)
    axes = (0, 2, 3)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int32 array, reduce over axis 0
    operand = np.array([[15, 240], [240, 15]], dtype=np.int32)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int64 array, reduce over axes (0, 1)
    operand = np.random.randint(-100, 100, size=(2, 2, 2), dtype=np.int64)
    axes = (0, 1)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D bool array, empty axes tuple
    operand = np.array([True, False], dtype=np.bool_)
    axes = ()
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int32 array, reduce over axes (1, 3)
    operand = np.random.randint(0, 10, size=(2, 2, 3, 2, 4), dtype=np.int32)
    axes = (1, 3)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D bool array, reduce over all axes
    operand = np.array([[True, True], [True, False]], dtype=np.bool_)
    axes = (0, 1)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reduce_or_2"] = jax_lax_reduce_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_or_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_or_2'.")


check_valid('jax.lax.reduce_or', generated_inputs['jax.lax.reduce_or_2'], lib="jax", suffix=2)
