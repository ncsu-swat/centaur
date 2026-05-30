
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def triu_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix, float32, k=0
    m = np.random.randn(5, 5).astype(np.float32)
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square matrix, int32, k=1
    m = np.random.randint(-10, 10, size=(4, 4)).astype(np.int32)
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D square matrix, float64, k=-1
    m = np.random.randn(6, 6).astype(np.float64)
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, float32, k=0
    m = np.random.randn(2, 3, 3).astype(np.float32)
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, int64, k=2
    m = np.random.randint(-100, 100, size=(3, 4, 4)).astype(np.int64)
    k = 2
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D non-square matrix (tall), float32, k=-2
    m = np.random.randn(8, 4).astype(np.float32)
    k = -2
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D non-square matrix (wide), float32, k=3
    m = np.random.randn(3, 7).astype(np.float32)
    k = 3
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, bool, k=-1
    m = np.random.choice([True, False], size=(2, 2, 4, 4))
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, float32, large positive k
    m = np.random.randn(5, 5).astype(np.float32)
    k = 10
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, int32, large negative k
    m = np.random.randint(-5, 5, size=(2, 5, 5)).astype(np.int32)
    k = -10
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
