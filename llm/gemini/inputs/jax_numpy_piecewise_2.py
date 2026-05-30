
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def piecewise_inputs():
    list_of_inputs = []

    # Input 1: 1D float array, 1 element-wise condition, 2 functions (1 condition + 1 default)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    condlist = np.array([[True, False, True, False, True]], dtype=bool)
    funclist = [2.0, 10.0]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    # Input 2: 1D int array, 1 element-wise condition
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    condlist = np.array([[True, True, False, False]], dtype=bool)
    funclist = [100, 200]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    # Input 3: 2D float array, 1 element-wise condition matching the 2D shape
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    condlist = np.array([[[True, False, True], [False, True, False]]], dtype=bool)
    funclist = [5.0, 0.0]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    # Input 4: 1D int64 array, 1 element-wise condition, single function (defaulting to 0)
    x = np.array([-1, 0, 1], dtype=np.int64)
    condlist = np.array([[True, False, True]], dtype=bool)
    funclist = [9]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    # Input 5: 1D float32 array, 2 element-wise conditions, 3 functions (2 conditions + 1 default)
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    condlist = np.array([
        [True, False, True, False, True],
        [False, True, False, True, False]
    ], dtype=bool)
    funclist = [1.0, 10.0, 100.0]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    # Input 6: 1D float64 array, 1 element-wise condition
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    condlist = np.array([[True, False, True, False]], dtype=bool)
    funclist = [-1.0, 1.0]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    # Input 7: 1D negative int32 array, 2 element-wise conditions
    x = np.array([-5, -4, -3, -2, -1], dtype=np.int32)
    condlist = np.array([
        [True, False, True, False, True],
        [False, True, False, True, False]
    ], dtype=bool)
    funclist = [10, -10, 0]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    # Input 8: 2D int32 array, 1 element-wise condition
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    condlist = np.array([[[False, True], [True, False]]], dtype=bool)
    funclist = [0, 100]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    # Input 9: 1D float32 array, 1 element-wise condition
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    condlist = np.array([[True, True, True, False, False]], dtype=bool)
    funclist = [0.0, 1.0]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    # Input 10: 1D int64 array, 1 element-wise condition
    x = np.array([10, 20, 30, 40], dtype=np.int64)
    condlist = np.array([[False, False, True, True]], dtype=bool)
    funclist = [5, 20]
    list_of_inputs.append({"x": x, "condlist": condlist, "funclist": funclist})

    return list_of_inputs

generated_inputs["jax.numpy.piecewise_2"] = piecewise_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.piecewise_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.piecewise_2'.")


check_valid('jax.numpy.piecewise', generated_inputs['jax.numpy.piecewise_2'], lib="jax", suffix=2)
