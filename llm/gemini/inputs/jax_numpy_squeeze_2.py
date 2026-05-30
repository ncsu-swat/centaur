
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def squeeze_inputs():
    list_of_inputs = []

    # Input 1: squeeze first dimension of size 1
    a = np.random.randn(1, 3, 1).astype(np.float32)
    axis = (0,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 2: squeeze last dimension of size 1
    a = np.random.randn(1, 3, 1).astype(np.float32)
    axis = (2,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 3: squeeze both first and last dimensions of size 1
    a = np.random.randn(1, 3, 1).astype(np.float32)
    axis = (0, 2)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 4: squeeze specific dimensions of a 4D tensor with all size 1
    a = np.random.randn(1, 1, 1, 1).astype(np.float32)
    axis = (0, 1)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 5: squeeze dimensions of size 1 in a 4D tensor
    a = np.random.randn(2, 1, 3, 1).astype(np.float32)
    axis = (1, 3)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 6: squeeze first dimension using negative axis index
    a = np.random.randn(1, 5, 5).astype(np.int32)
    axis = (-3,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 7: squeeze last dimension using negative axis index
    a = np.random.randn(5, 5, 1).astype(np.int32)
    axis = (-1,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 8: squeeze multiple dimensions using negative axis indices
    a = np.random.randn(1, 1, 5, 1, 1).astype(np.float64)
    axis = (-5, -4, -2, -1)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 9: squeeze subset of dimensions using negative axis indices
    a = np.random.randn(1, 1, 5, 1, 1).astype(np.float64)
    axis = (-4, -2)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 10: squeeze multiple dimensions of size 1 from 5D tensor
    a = np.random.randn(1, 2, 1, 3, 1).astype(np.float32)
    axis = (0, 2, 4)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    # Input 11: 2D array, squeeze only available unit axis
    a = np.random.randn(1, 10).astype(np.float32)
    axis = (0,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.squeeze_2"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.squeeze_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.squeeze_2'.")


check_valid('jax.numpy.squeeze', generated_inputs['jax.numpy.squeeze_2'], lib="jax", suffix=2)
