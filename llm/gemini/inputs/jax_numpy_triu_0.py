
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def triu_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float square matrix, k=0
    m = np.arange(9, dtype=np.float32).reshape(3, 3)
    k = 0
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 2: 2D rectangular matrix (more rows than columns), k=1
    m = np.random.randn(5, 3).astype(np.float64)
    k = 1
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 3: 2D rectangular matrix (more columns than rows), negative k
    m = np.arange(12, dtype=np.int32).reshape(3, 4)
    k = -1
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 4: 3D batched tensor, k=0
    m = np.random.randn(2, 4, 4).astype(np.float32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 5: 3D batched tensor with negative k
    m = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    k = -2
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 6: 4D tensor, positive k
    m = np.random.randn(2, 3, 5, 5).astype(np.float32)
    k = 2
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 7: Boolean array, k=0
    m = np.array(
        [[True, False, True], [False, True, False], [True, True, True]],
        dtype=bool,
    )
    k = 0
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 8: Complex array, negative k
    m = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex64
    )
    k = -1
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 9: Large k exceeding matrix dimensions (results in all zeros)
    m = np.ones((3, 3), dtype=np.float32)
    k = 10
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 10: Negative k covering entire matrix (leaves full matrix)
    m = np.ones((3, 3), dtype=np.int32)
    k = -10
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    # Input 11: 2x2 integer matrix with k=0
    m = np.array([[1, 2], [3, 4]], dtype=np.int16)
    k = 0
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    return list_of_inputs


generated_inputs["jax.numpy.triu"] = triu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.triu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.triu'.")


check_valid('jax.numpy.triu', generated_inputs['jax.numpy.triu'], lib="jax", suffix=0)
