
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: Basic integer division with 1D int32 array
    x1 = 10
    x2 = np.array([3, 4, 7], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 2: Negative dividend, 1D int32 array with mixed signs
    x1 = -10
    x2 = np.array([-3, 4, -7], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 3: Positive dividend, 2D float32 array
    x1 = 5
    x2 = np.array([[2.0, 1.5], [0.5, -2.5]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 4: Zero dividend, 3D int64 array
    x1 = 0
    x2 = np.random.randint(1, 10, size=(2, 2, 2)).astype(np.int64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 5: Large dividend, 1D float64 array
    x1 = 100
    x2 = np.array([1.2, 3.4, 5.6], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 6: Negative dividend, 2D int32 array
    x1 = -50
    x2 = np.array([[3, 6], [-2, -8]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 7: Small dividend, 4D int32 array
    x1 = 1
    x2 = (np.ones((2, 2, 2, 2)) * 2).astype(np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 8: Positive dividend, 1D int64 array
    x1 = 42
    x2 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 9: Large negative dividend, 2D float32 array
    x1 = -123
    x2 = np.array([[100.0, -200.0], [50.0, -50.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 10: Positive dividend, 1D int64 array with mixed signs
    x1 = 8
    x2 = np.array([2, -3, 4], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.floor_divide_4"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_divide_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_divide_4'.")


check_valid('jax.numpy.floor_divide', generated_inputs['jax.numpy.floor_divide_4'], lib="jax", suffix=4)
