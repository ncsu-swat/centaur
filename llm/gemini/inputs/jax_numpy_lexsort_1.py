
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lexsort_inputs():
    list_of_inputs = []

    # Input 1: 2D array of integers (sequence of two 1D keys), default axis (-1)
    keys = np.array([[4, 2, 3, 2, 5], [2, 1, 1, 2, 2]], dtype=np.int32)
    input_dict = {"keys": keys, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array of floats, with negative values, axis = 0
    keys = np.array([[-1.5, 2.3, 0.0, -2.3], [0.5, -0.5, 1.2, 1.2]], dtype=np.float32)
    input_dict = {"keys": keys, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of int64 with a single row (equivalent to argsort)
    keys = np.array([[10, 5, 8, 1, 3]], dtype=np.int64)
    input_dict = {"keys": keys, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array of integers, sorting along the last axis (-1)
    keys = np.random.randint(0, 10, size=(2, 2, 4))
    input_dict = {"keys": keys, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of integers, sorting along axis 0
    keys = np.random.randint(0, 10, size=(2, 2, 4))
    input_dict = {"keys": keys, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of booleans
    keys = np.array([
        [True, False, True, False],
        [False, True, False, True],
        [True, True, False, False]
    ], dtype=bool)
    input_dict = {"keys": keys, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D random integer array, sorting along axis 2
    keys = np.random.randint(-5, 5, size=(2, 2, 3, 4))
    input_dict = {"keys": keys, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float array, sorting along axis 1
    keys = np.random.randn(2, 2, 2, 3).astype(np.float32)
    input_dict = {"keys": keys, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 2D float64 keys
    keys = np.array([np.linspace(10, 0, 100), np.linspace(0, 10, 100)], dtype=np.float64)
    input_dict = {"keys": keys, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, sorting along axis -2
    keys = np.random.randint(0, 10, size=(2, 2, 2, 2, 3))
    input_dict = {"keys": keys, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.lexsort_1"] = lexsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.lexsort_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.lexsort_1'.")


check_valid('jax.numpy.lexsort', generated_inputs['jax.numpy.lexsort_1'], lib="jax", suffix=1)
