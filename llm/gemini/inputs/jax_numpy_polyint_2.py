
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polyint_inputs():
    list_of_inputs = []

    # Input 1: Float32, m=1
    p = np.array([12.0, 12.0, 6.0], dtype=np.float32)
    m = 1
    k = np.array([4.0], dtype=np.float32)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, m=2
    p = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    m = 2
    k = np.array([5.0, 6.0], dtype=np.float64)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int32, m=1
    p = np.array([1, -2, 3], dtype=np.int32)
    m = 1
    k = np.array([-1], dtype=np.int32)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float16, m=3
    p = np.array([2.0, 0.0, -1.0, 5.0], dtype=np.float16)
    m = 3
    k = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative values, m=1
    p = np.array([-100.0, -50.0, -25.0], dtype=np.float32)
    m = 1
    k = np.array([-10.0], dtype=np.float32)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large array, m=4
    p = np.arange(10, dtype=np.float32)
    m = 4
    k = np.zeros(4, dtype=np.float32)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zeros, m=2
    p = np.zeros(5, dtype=np.float32)
    m = 2
    k = np.array([1.5, -2.5], dtype=np.float32)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64, m=1
    p = np.array([1+2j, 3+4j], dtype=np.complex64)
    m = 1
    k = np.array([5+6j], dtype=np.complex64)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int64, m=2
    p = np.array([10, 20, 30], dtype=np.int64)
    m = 2
    k = np.array([1, 2], dtype=np.int64)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element (constant polynomial), m=1
    p = np.array([5.0], dtype=np.float32)
    m = 1
    k = np.array([2.0], dtype=np.float32)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.polyint_2"] = polyint_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyint_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyint_2'.")


check_valid('jax.numpy.polyint', generated_inputs['jax.numpy.polyint_2'], lib="jax", suffix=2)
