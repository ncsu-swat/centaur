
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tril_indices_from_inputs():
    list_of_inputs = []

    # Input 1: Square matrix with default-like k=0
    arr = np.zeros((3, 3), dtype=np.float32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 2: Square matrix with positive k
    arr = np.ones((4, 4), dtype=np.int32)
    k = 1
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 3: Square matrix with negative k
    arr = np.arange(25, dtype=np.int64).reshape((5, 5))
    k = -2
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 4: Rectangular matrix (more columns than rows) with positive k
    arr = np.empty((3, 6), dtype=np.float64)
    k = 2
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 5: Rectangular matrix (more rows than columns) with negative k
    arr = np.zeros((6, 3), dtype=np.bool_)
    k = -1
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 6: 1x1 matrix with k=0
    arr = np.array([[42.0]], dtype=np.float32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 7: Complex matrix with k larger than dimension
    arr = np.ones((3, 3), dtype=np.complex64)
    k = 5
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 8: Matrix with negative k smaller than -dimension
    arr = np.zeros((4, 4), dtype=np.float32)
    k = -10
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 9: Non-square matrix with k=0
    arr = np.ones((2, 5), dtype=np.uint8)
    k = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 10: Matrix with 0 rows or columns
    arr = np.zeros((0, 0), dtype=np.int32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    # Input 11: Larger matrix with positive k
    arr = np.eye(10, dtype=np.float32)
    k = 3
    list_of_inputs.append(copy.deepcopy({"arr": arr, "k": k}))

    return list_of_inputs


generated_inputs["jax.numpy.tril_indices_from"] = tril_indices_from_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tril_indices_from' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tril_indices_from'.")


check_valid('jax.numpy.tril_indices_from', generated_inputs['jax.numpy.tril_indices_from'], lib="jax", suffix=0)
