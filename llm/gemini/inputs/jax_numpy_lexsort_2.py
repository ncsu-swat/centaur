
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lexsort_inputs():
    list_of_inputs = []

    # Input 1: 2 keys of 1D integer arrays, axis=-1
    keys = np.array([[4, 2, 3, 2, 5], [2, 1, 1, 2, 2]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": -1}))

    # Input 2: 1 key of 1D float array, axis=0
    keys = np.array([[3.5, -1.2, 0.0, 4.8, -2.1]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": 0}))

    # Input 3: 3 keys of 1D integer arrays, axis=-1
    keys = np.array(
        [
            [1, 5, 2, 3, 1, 4],
            [2, 2, 1, 1, 3, 3],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": -1}))

    # Input 4: 2 keys of 2D integer arrays, axis=0
    keys = np.array(
        [
            [[2, 4, 2, 3], [3, 1, 2, 2]],
            [[1, 2, 1, 3], [2, 1, 2, 1]],
        ],
        dtype=np.int32,
    )
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": 0}))

    # Input 5: 2 keys of 2D float arrays, axis=1
    keys = np.array(
        [
            [[1.1, -2.2, 3.3], [4.4, 0.0, -1.5]],
            [[0.5, 0.5, -1.0], [2.0, -3.0, 1.0]],
        ],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": 1}))

    # Input 6: 2 keys of 2D boolean arrays, axis=-1
    keys = np.array(
        [
            [[True, False, True], [False, True, False]],
            [[False, True, True], [True, False, False]],
        ],
        dtype=np.bool_,
    )
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": -1}))

    # Input 7: 2 keys of 2D arrays, axis=-2
    keys = np.array(
        [
            [[5, 2], [9, 1], [3, 7]],
            [[1, 4], [1, 2], [2, 2]],
        ],
        dtype=np.int32,
    )
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": -2}))

    # Input 8: 3 keys of 3D float arrays, axis=-1
    keys = np.random.randn(3, 2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": -1}))

    # Input 9: 2 keys of 3D integer arrays, axis=1
    keys = np.random.randint(-10, 10, size=(2, 4, 3, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": 1}))

    # Input 10: 4 keys of 1D integer arrays, axis=0
    keys = np.array(
        [
            [10, 20, 10, 40, 30],
            [1, 1, 2, 2, 1],
            [5, 4, 3, 2, 1],
            [0, 0, 1, 1, 0],
        ],
        dtype=np.int32,
    )
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": 0}))

    # Input 11: 2 keys of 1D arrays with negative values and duplicates, axis=-1
    keys = np.array(
        [
            [-5, -1, -5, 0, 2, -1],
            [-2, -2, -3, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    list_of_inputs.append(copy.deepcopy({"keys": keys, "axis": -1}))

    return list_of_inputs


generated_inputs["jax.numpy.lexsort_2"] = lexsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.lexsort_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.lexsort_2'.")


check_valid('jax.numpy.lexsort', generated_inputs['jax.numpy.lexsort_2'], lib="jax", suffix=2)
