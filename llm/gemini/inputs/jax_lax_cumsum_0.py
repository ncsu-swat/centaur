
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cumsum_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, reverse=False
    operand = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {'operand': operand, 'axis': 0, 'reverse': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with negative values, axis 0, reverse=True
    operand = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    input_dict = {'operand': operand, 'axis': 0, 'reverse': True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 1, reverse=False
    operand = np.random.randn(3, 4).astype(np.float64)
    input_dict = {'operand': operand, 'axis': 1, 'reverse': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, axis 2, reverse=True
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {'operand': operand, 'axis': 2, 'reverse': True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, axis 0, reverse=False
    operand = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int64)
    input_dict = {'operand': operand, 'axis': 0, 'reverse': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, axis 1, reverse=True
    operand = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {'operand': operand, 'axis': 1, 'reverse': True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float16 array, axis 0, reverse=False
    operand = np.array([-0.5, 1.5, -2.5, 3.5], dtype=np.float16)
    input_dict = {'operand': operand, 'axis': 0, 'reverse': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32 array, axis 0, reverse=True
    operand = np.random.randint(-100, 100, size=(5, 5)).astype(np.int32)
    input_dict = {'operand': operand, 'axis': 0, 'reverse': True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float64 array, axis 3, reverse=False
    operand = np.random.randn(2, 2, 2, 3, 2).astype(np.float64)
    input_dict = {'operand': operand, 'axis': 3, 'reverse': False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int32 array, axis 1, reverse=True
    operand = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32)
    input_dict = {'operand': operand, 'axis': 1, 'reverse': True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.cumsum"] = cumsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cumsum'.")


check_valid('jax.lax.cumsum', generated_inputs['jax.lax.cumsum'], lib="jax", suffix=0)
