
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, y=True
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = True
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 2: 2D int32 array, y=False
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = False
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 3: 3D int64 array, y=True
    x = np.arange(8, dtype=np.int64).reshape((2, 2, 2))
    y = True
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 4: 1D bool array, y=False
    x = np.array([True, False, True], dtype=np.bool_)
    y = False
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 5: 2D bool array, y=True
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = True
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 6: 1D negative int32 array, y=True
    x = np.array([-1, -2, -3, -4], dtype=np.int32)
    y = True
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 7: 2D negative int64 array, y=False
    x = np.array([[-10, -20], [-30, -40]], dtype=np.int64)
    y = False
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 8: 4D int32 array, y=True
    x = np.ones((2, 2, 2, 2), dtype=np.int32)
    y = True
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 9: 0D int32 array, y=False
    x = np.array(5, dtype=np.int32)
    y = False
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 10: 1D int64 array, y=True
    x = np.array([1024, 2048, 4096], dtype=np.int64)
    y = True
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_or_3"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_or_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_or_3'.")


check_valid('jax.numpy.bitwise_or', generated_inputs['jax.numpy.bitwise_or_3'], lib="jax", suffix=3)
