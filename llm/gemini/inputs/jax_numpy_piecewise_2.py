
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def piecewise_inputs():
    list_of_inputs = []

    # Input 1: 1D float array with single boolean condition and constant scalar functions
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    condlist = (x < 0.0).reshape(1, -1)
    funclist = [0.0, 1.0]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

    # Input 2: 1D int array with multiple conditions and constant scalar functions
    x = np.array([-5, -2, 0, 3, 7], dtype=np.int32)
    condlist = np.stack([x < 0, (x >= 0) & (x < 5), x >= 5], axis=0)
    funclist = [-1, 0, 1]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

    # Input 3: 2D float array, conditions shaped (num_cond, *x.shape)
    x = np.array([[-1.5, 2.5], [3.0, -0.5]], dtype=np.float32)
    condlist = np.stack([x < 0.0, x >= 0.0], axis=0)
    funclist = [0.0, 2.0]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

    # Input 4: 1D float64 array with default value (len(funclist) == len(condlist) + 1)
    x = np.linspace(-3.0, 3.0, 7, dtype=np.float64)
    condlist = np.stack([x < -1.0, x > 1.0], axis=0)
    funclist = [10.0, -10.0, 0.0]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

    # Input 5: 1D scalar boolean conditions tensor applying to whole array
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    condlist = np.array([False, True])
    funclist = [2.0, 10.0]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

    # Input 6: 3D float array
    x = np.arange(-4, 4, dtype=np.float32).reshape(2, 2, 2)
    condlist = np.stack([x < 0, x >= 0], axis=0)
    funclist = [-1.0, 1.0]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

    # Input 7: 1D int64 array
    x = np.array([-10, 0, 10, 20], dtype=np.int64)
    condlist = np.stack([x < 0, x == 0, x > 0], axis=0)
    funclist = [-1, 0, 1]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

    # Input 8: 2D int32 array with default condition
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    condlist = np.stack([x % 2 == 0], axis=0)
    funclist = [2, 3]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

    # Input 9: 1D array of single element
    x = np.array([-3.14], dtype=np.float32)
    condlist = np.stack([x < 0.0, x >= 0.0], axis=0)
    funclist = [-1.0, 1.0]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

    # Input 10: 4D float32 array
    x = np.ones((1, 2, 2, 1), dtype=np.float32) * -1.0
    condlist = np.stack([x < 0.0], axis=0)
    funclist = [5.0, 0.0]
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "condlist": condlist, "funclist": funclist}
    )

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
