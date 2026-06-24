
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gather_inputs():
    list_of_inputs = []

    # To completely avoid any potential (5,) inhomogeneous shape issues,
    # we use a plain tuple of 3 homogeneous tuples for dimension_numbers,
    # which is automatically converted to GatherDimensionNumbers by JAX.
    # No variable in the dictionary will have a length of 5.
    dimension_numbers = ((1,), (0,), (0,))
    slice_sizes = [1, 4]

    # Input 1
    operand = np.random.randn(4, 4).astype(np.float32)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operand = np.random.randn(4, 4).astype(np.float32)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': "fill",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    operand = np.random.randn(4, 4).astype(np.float64)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int64)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "promise_in_bounds",
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    operand = np.random.randint(0, 100, size=(4, 4)).astype(np.int32)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': "promise_in_bounds",
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operand = np.random.randn(4, 4).astype(np.float32)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    operand = np.random.randn(4, 4).astype(np.float32)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': "fill",
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operand = np.random.randn(4, 4).astype(np.float64)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int64)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': "clip",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    operand = np.random.randint(0, 50, size=(4, 4)).astype(np.int32)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "promise_in_bounds",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    operand = np.random.randn(4, 4).astype(np.float32)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    operand = np.random.randn(4, 4).astype(np.float32)
    start_indices = np.random.randint(0, 4, size=(2, 1)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': "fill",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_6"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_6'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_6'], lib="jax", suffix=6)
