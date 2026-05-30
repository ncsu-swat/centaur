
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tril_indices_from_inputs():
    list_of_inputs = []

    # Input 1: Square matrix, k=0
    arr = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"arr": arr, "k": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Square matrix, k > 0
    arr = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"arr": arr, "k": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Square matrix, k < 0
    arr = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"arr": arr, "k": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular matrix (wide), k=0
    arr = np.random.randn(3, 5).astype(np.float32)
    input_dict = {"arr": arr, "k": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrix (tall), k=0
    arr = np.random.randn(5, 3).astype(np.float32)
    input_dict = {"arr": arr, "k": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rectangular matrix, k > 0
    arr = np.random.randn(4, 6).astype(np.float32)
    input_dict = {"arr": arr, "k": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rectangular matrix, k < 0
    arr = np.random.randn(6, 4).astype(np.float32)
    input_dict = {"arr": arr, "k": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger square matrix, k large negative
    arr = np.random.randn(10, 10).astype(np.float32)
    input_dict = {"arr": arr, "k": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger square matrix, k large positive
    arr = np.random.randn(10, 10).astype(np.float32)
    input_dict = {"arr": arr, "k": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Minimal size 1x1, k=0
    arr = np.random.randn(1, 1).astype(np.float32)
    input_dict = {"arr": arr, "k": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Integer array with out-of-bounds positive k
    arr = np.zeros((4, 4), dtype=np.int32)
    input_dict = {"arr": arr, "k": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Boolean array with out-of-bounds negative k
    arr = np.ones((4, 4), dtype=bool)
    input_dict = {"arr": arr, "k": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
