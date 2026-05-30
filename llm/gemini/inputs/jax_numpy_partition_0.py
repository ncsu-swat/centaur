
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def partition_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, positive kth, default-like axis
    a = np.array([6.5, 8.1, -4.2, 3.0, 1.1, 9.2, 7.0, 5.5, 2.3, 3.0], dtype=np.float32)
    input_dict = {"a": a, "kth": 4, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, partition along axis 0
    a = np.array([[5, 2, 9], [1, 8, 3], [4, 7, 6]], dtype=np.int32)
    input_dict = {"a": a, "kth": 1, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, partition along axis 1
    a = np.random.randn(5, 5).astype(np.float64)
    input_dict = {"a": a, "kth": 2, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int16 array, partition along axis 2
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int16)
    input_dict = {"a": a, "kth": 1, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D int8 array
    a = np.array([1, 0, 1, 0, 0, 1, -1, 2], dtype=np.int8)
    input_dict = {"a": a, "kth": 2, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, partition with positive kth along axis -1
    a = np.random.randn(4, 6).astype(np.float32)
    input_dict = {"a": a, "kth": 3, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, partition along axis 2
    a = np.random.randn(2, 3, 5, 2).astype(np.float32)
    input_dict = {"a": a, "kth": 3, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array of int64 with duplicate elements
    a = np.array([10, 10, 5, 5, 1, 1, 8, 8], dtype=np.int64)
    input_dict = {"a": a, "kth": 5, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array with small size, partition along axis 1
    a = np.random.uniform(-5, 5, (2, 4, 3)).astype(np.float32)
    input_dict = {"a": a, "kth": 0, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, larger size
    a = np.random.randn(100).astype(np.float32)
    input_dict = {"a": a, "kth": 50, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.partition"] = partition_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.partition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.partition'.")


check_valid('jax.numpy.partition', generated_inputs['jax.numpy.partition'], lib="jax", suffix=0)
