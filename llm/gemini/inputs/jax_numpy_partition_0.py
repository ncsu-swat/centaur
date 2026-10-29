
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def partition_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    a = np.array([6, 8, 4, 3, 1, 9, 7, 5, 2, 3], dtype=np.int32)
    kth = 4
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 2: 1D float32 array with negative values
    a = np.array([-3.5, 1.2, -0.1, 8.4, -9.9, 0.0, 4.2], dtype=np.float32)
    kth = 2
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 3: 2D float64 array, partition along columns
    a = np.random.randn(5, 8).astype(np.float64)
    kth = 0
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 4: 2D int64 array, partition along rows
    a = np.random.randint(-100, 100, size=(10, 6), dtype=np.int64)
    kth = 3
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 5: 3D float32 array, partition along last axis
    a = np.random.randn(3, 4, 12).astype(np.float32)
    kth = 5
    axis = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 6: 3D int32 array with negative kth
    a = np.random.randint(-50, 50, size=(2, 7, 4), dtype=np.int32)
    kth = -2
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 7: 4D float32 array, partition along negative axis
    a = np.random.randn(2, 3, 6, 5).astype(np.float32)
    kth = 1
    axis = -2
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 8: 2D float16 array
    a = np.random.randn(8, 15).astype(np.float16)
    kth = 7
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 9: 1D float64 array with maximum valid kth index
    a = np.array([5.0, 2.0, 9.0, 1.0, 7.0], dtype=np.float64)
    kth = 4
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 10: 2D float32 array with zero kth
    a = np.random.randn(4, 6).astype(np.float32)
    kth = 0
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 11: 1D int32 array with negative kth (-1 selects largest element)
    a = np.array([10, -5, 3, 20, 0, -1], dtype=np.int32)
    kth = -1
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

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
