
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def triu_indices_from_inputs():
    list_of_inputs = []

    # Input 1: 3x3 float32, k=0
    arr = np.random.randn(3, 3).astype(np.float32)
    k = 0
    list_of_inputs.append({"arr": arr, "k": k})

    # Input 2: 4x4 int32, k=1
    arr = np.random.randint(-10, 10, size=(4, 4)).astype(np.int32)
    k = 1
    list_of_inputs.append({"arr": arr, "k": k})

    # Input 3: 5x5 float64, k=-1
    arr = np.random.randn(5, 5).astype(np.float64)
    k = -1
    list_of_inputs.append({"arr": arr, "k": k})

    # Input 4: Rectangular 2x3 float32, k=2
    arr = np.random.randn(2, 3).astype(np.float32)
    k = 2
    list_of_inputs.append({"arr": arr, "k": k})

    # Input 5: Rectangular 6x2 int32, k=-2
    arr = np.random.randint(-10, 10, size=(6, 2)).astype(np.int32)
    k = -2
    list_of_inputs.append({"arr": arr, "k": k})

    # Input 6: 10x10 bool, k=0
    arr = np.random.choice([True, False], size=(10, 10))
    k = 0
    list_of_inputs.append({"arr": arr, "k": k})

    # Input 7: 1x1 float32, k=0
    arr = np.random.randn(1, 1).astype(np.float32)
    k = 0
    list_of_inputs.append({"arr": arr, "k": k})

    # Input 8: Large square 8x8 float32, k=5
    arr = np.random.randn(8, 8).astype(np.float32)
    k = 5
    list_of_inputs.append({"arr": arr, "k": k})

    # Input 9: Rectangular 5x7 int16, k=-3
    arr = np.random.randint(-100, 100, size=(5, 7)).astype(np.int16)
    k = -3
    list_of_inputs.append({"arr": arr, "k": k})

    # Input 10: 100x100 float32, k=-50
    arr = np.random.randn(100, 100).astype(np.float32)
    k = -50
    list_of_inputs.append({"arr": arr, "k": k})

    return list_of_inputs

generated_inputs["jax.numpy.triu_indices_from"] = triu_indices_from_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.triu_indices_from' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.triu_indices_from'.")


check_valid('jax.numpy.triu_indices_from', generated_inputs['jax.numpy.triu_indices_from'], lib="jax", suffix=0)
