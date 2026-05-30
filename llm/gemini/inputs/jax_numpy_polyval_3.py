
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polyval_inputs():
    list_of_inputs = []

    # Input 1: Standard float32 array, small positive integer x, default-like unroll
    p1 = np.array([2.0, 5.0, 1.0], dtype=np.float32)
    input_dict = {"p": p1, "x": 3, "unroll": 16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer array, negative integer x, different unroll
    p2 = np.array([1, -2, 0, 4], dtype=np.int32)
    input_dict = {"p": p2, "x": -2, "unroll": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 array, large x, large unroll
    p3 = np.array([1.5, -3.2, 0.5, 9.1], dtype=np.float64)
    input_dict = {"p": p3, "x": 10, "unroll": 128}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1-element array (constant), x=0, unroll=1
    p4 = np.array([5.0], dtype=np.float32)
    input_dict = {"p": p4, "x": 0, "unroll": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Long array, negative x, unroll=32
    p5 = np.random.randn(20).astype(np.float32)
    input_dict = {"p": p5, "x": -1, "unroll": 32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large integer coefficients, x=5, unroll=64
    p6 = np.array([1000, 2000, 3000], dtype=np.int64)
    input_dict = {"p": p6, "x": 5, "unroll": 64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero array, large x, unroll=16
    p7 = np.zeros(10, dtype=np.float32)
    input_dict = {"p": p7, "x": 100, "unroll": 16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All ones array, x=-5, unroll=4
    p8 = np.ones(5, dtype=np.float32)
    input_dict = {"p": p8, "x": -5, "unroll": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Alternating signs array, x=1, unroll=256
    p9 = np.array([-1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    input_dict = {"p": p9, "x": 1, "unroll": 256}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large random float64 array, x=2, unroll=16
    p10 = np.random.uniform(-10, 10, size=(50,)).astype(np.float64)
    input_dict = {"p": p10, "x": 2, "unroll": 16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.polyval_3"] = polyval_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyval_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyval_3'.")


check_valid('jax.numpy.polyval', generated_inputs['jax.numpy.polyval_3'], lib="jax", suffix=3)
