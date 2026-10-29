
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np


def lexsort_inputs():
    list_of_inputs = []

    # Input 1: 2D integer array (2 keys, length 5), axis=-1
    keys = np.array([[4, 2, 3, 2, 5], [2, 1, 1, 2, 2]], dtype=np.int32)
    list_of_inputs.append({"keys": keys, "axis": -1})

    # Input 2: 2D integer array (2 keys, length 6), axis=0
    keys = np.array(
        [[10, -5, 0, 3, 7, -2], [1, 1, 2, 2, 0, 0]], dtype=np.int64
    )
    list_of_inputs.append({"keys": keys, "axis": 0})

    # Input 3: 2D float array with negative values, axis=-1
    keys = np.array(
        [[-1.5, 2.3, -0.5, 0.0, 2.3], [3.0, -1.0, -1.0, 0.5, 2.0]],
        dtype=np.float32,
    )
    list_of_inputs.append({"keys": keys, "axis": -1})

    # Input 4: 2D float64 array, axis=0
    keys = np.array(
        [[0.5, -1.2, 3.4, -0.1], [1.0, 1.0, -2.0, -2.0]], dtype=np.float64
    )
    list_of_inputs.append({"keys": keys, "axis": 0})

    # Input 5: 3 keys of 1D arrays represented as a (3, 7) array, axis=-1
    keys = np.random.randint(-10, 10, size=(3, 7), dtype=np.int32)
    list_of_inputs.append({"keys": keys, "axis": -1})

    # Input 6: 3D integer array (2 keys of shape (3, 4)), axis=0
    keys = np.random.randint(0, 10, size=(2, 3, 4), dtype=np.int32)
    list_of_inputs.append({"keys": keys, "axis": 0})

    # Input 7: 3D float32 array (2 keys of shape (4, 5)), axis=1
    keys = np.random.randn(2, 4, 5).astype(np.float32)
    list_of_inputs.append({"keys": keys, "axis": 1})

    # Input 8: 3D integer array (3 keys of shape (4, 6)), axis=-1
    keys = np.random.randint(-20, 20, size=(3, 4, 6), dtype=np.int64)
    list_of_inputs.append({"keys": keys, "axis": -1})

    # Input 9: 4D float32 array (2 keys of shape (2, 3, 4)), axis=2
    keys = np.random.randn(2, 2, 3, 4).astype(np.float32)
    list_of_inputs.append({"keys": keys, "axis": 2})

    # Input 10: 4D integer array (2 keys of shape (2, 3, 5)), axis=-2
    keys = np.random.randint(-5, 5, size=(2, 2, 3, 5), dtype=np.int32)
    list_of_inputs.append({"keys": keys, "axis": -2})

    # Input 11: 2D boolean array (2 keys of length 6), axis=-1
    keys = np.array(
        [
            [True, False, True, False, True, False],
            [False, False, True, True, False, True],
        ],
        dtype=np.bool_,
    )
    list_of_inputs.append({"keys": keys, "axis": -1})

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
