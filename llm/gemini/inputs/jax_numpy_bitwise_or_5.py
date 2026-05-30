
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_bitwise_or_inputs():
    list_of_inputs = []

    # Input 1
    x = True
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2
    x = False
    y = np.array([[-1, 2], [3, -4]], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3
    x = True
    y = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4
    x = False
    y = np.random.randint(-50, 50, size=(2, 2, 3)).astype(np.int16)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5
    x = True
    y = np.random.randint(-100, 100, size=(4, 4)).astype(np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6
    x = False
    y = np.array([-128, -1, 0, 127], dtype=np.int8)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7
    x = True
    y = np.random.randint(-1000, 1000, size=(1, 2, 2, 1)).astype(np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8
    x = True
    y = np.array([False], dtype=bool)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9
    x = False
    y = np.random.randint(-10000, 10000, size=(5, 5)).astype(np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10
    x = True
    y = np.array([123456789, 987654321], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_or_5"] = jax_numpy_bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_or_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_or_5'.")


check_valid('jax.numpy.bitwise_or', generated_inputs['jax.numpy.bitwise_or_5'], lib="jax", suffix=5)
