
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix, main diagonal
    a = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"a": a, "offset": 0, "axis1": 0, "axis2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square matrix, positive offset
    a = np.random.randint(-10, 10, size=(4, 4)).astype(np.int32)
    input_dict = {"a": a, "offset": 1, "axis1": 0, "axis2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D square matrix, negative offset, float64
    a = np.random.randn(5, 5).astype(np.float64)
    input_dict = {"a": a, "offset": -2, "axis1": 0, "axis2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, axes 0 and 1, offset 0
    a = np.random.randn(3, 3, 4).astype(np.float32)
    input_dict = {"a": a, "offset": 0, "axis1": 0, "axis2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor, axes 0 and 2, offset 2
    a = np.random.randn(3, 5, 4).astype(np.float32)
    input_dict = {"a": a, "offset": 2, "axis1": 0, "axis2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor, negative axes, negative offset
    a = np.random.randn(4, 4, 4).astype(np.float32)
    input_dict = {"a": a, "offset": -1, "axis1": -2, "axis2": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor, offset 1
    a = np.random.randint(-100, 100, size=(2, 3, 4, 5)).astype(np.int32)
    input_dict = {"a": a, "offset": 1, "axis1": 1, "axis2": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D non-square matrix, offset 0
    a = np.random.randn(3, 5).astype(np.float32)
    input_dict = {"a": a, "offset": 0, "axis1": 0, "axis2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D non-square matrix, negative offset
    a = np.random.randn(6, 3).astype(np.float64)
    input_dict = {"a": a, "offset": -3, "axis1": 0, "axis2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D tensor, offset 0
    a = np.random.randn(2, 2, 3, 3, 2).astype(np.float32)
    input_dict = {"a": a, "offset": 0, "axis1": 2, "axis2": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.diagonal"] = diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diagonal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diagonal'.")


check_valid('jax.numpy.diagonal', generated_inputs['jax.numpy.diagonal'], lib="jax", suffix=0)
