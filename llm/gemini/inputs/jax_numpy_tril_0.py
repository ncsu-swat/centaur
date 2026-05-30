
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tril_inputs():
    list_of_inputs = []

    # Input 1: Standard 2D float32 square matrix, k=0
    m = np.random.randn(5, 5).astype(np.float32)
    k = 0
    list_of_inputs.append({"m": m, "k": k})

    # Input 2: 2D int32 matrix, positive k
    m = np.random.randint(-10, 10, size=(4, 6)).astype(np.int32)
    k = 2
    list_of_inputs.append({"m": m, "k": k})

    # Input 3: 2D float64 matrix, negative k
    m = np.random.randn(6, 4).astype(np.float64)
    k = -2
    list_of_inputs.append({"m": m, "k": k})

    # Input 4: 3D float32 batch matrix, k=0
    m = np.random.randn(2, 4, 4).astype(np.float32)
    k = 0
    list_of_inputs.append({"m": m, "k": k})

    # Input 5: 3D int64 batch matrix, positive k
    m = np.random.randint(-100, 100, size=(3, 5, 5)).astype(np.int64)
    k = 1
    list_of_inputs.append({"m": m, "k": k})

    # Input 6: 4D float32 batch matrix, negative k
    m = np.random.randn(2, 2, 3, 3).astype(np.float32)
    k = -1
    list_of_inputs.append({"m": m, "k": k})

    # Input 7: 2D boolean matrix, k=0
    m = np.random.choice([True, False], size=(5, 5))
    k = 0
    list_of_inputs.append({"m": m, "k": k})

    # Input 8: Tall 2D matrix (rows > cols), positive k
    m = np.random.randn(10, 3).astype(np.float32)
    k = 3
    list_of_inputs.append({"m": m, "k": k})

    # Input 9: Wide 2D matrix (cols > rows), negative k
    m = np.random.randn(3, 10).astype(np.float32)
    k = -3
    list_of_inputs.append({"m": m, "k": k})

    # Input 10: 2D float32 matrix, large positive k (no-op, entire matrix retained)
    m = np.random.randn(4, 4).astype(np.float32)
    k = 10
    list_of_inputs.append({"m": m, "k": k})

    # Input 11: 2D float32 matrix, large negative k (zeroes out entire matrix)
    m = np.random.randn(4, 4).astype(np.float32)
    k = -10
    list_of_inputs.append({"m": m, "k": k})

    return list_of_inputs

generated_inputs["jax.numpy.tril"] = tril_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tril' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tril'.")


check_valid('jax.numpy.tril', generated_inputs['jax.numpy.tril'], lib="jax", suffix=0)
