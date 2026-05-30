
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polyint_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D array, order 1, constant 0
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = 1
    k = 0
    list_of_inputs.append({"p": p, "m": m, "k": k})

    # Input 2: float32 1D array, order 1, positive constant
    p = np.array([12.0, 12.0, 6.0], dtype=np.float32)
    m = 1
    k = 4
    list_of_inputs.append({"p": p, "m": m, "k": k})

    # Input 3: float64 1D array with negative values, order 1, negative constant
    p = np.array([-1.0, 0.0, 5.0], dtype=np.float64)
    m = 1
    k = -1
    list_of_inputs.append({"p": p, "m": m, "k": k})

    # Input 4: int32 1D array, order 2, constant 0
    p = np.array([1, 2, 3, 4], dtype=np.int32)
    m = 2
    k = 0
    list_of_inputs.append({"p": p, "m": m, "k": k})

    # Input 5: int64 1D array, order 2, positive constant
    p = np.array([10, -5, 2], dtype=np.int64)
    m = 2
    k = 5
    list_of_inputs.append({"p": p, "m": m, "k": k})

    # Input 6: Longer float32 array, order 1, constant 2
    p = np.array([0.5, -0.2, 0.1, 9.0], dtype=np.float32)
    m = 1
    k = 2
    list_of_inputs.append({"p": p, "m": m, "k": k})

    # Input 7: Short float64 array, order 3, negative constant
    p = np.array([3.0, 6.0], dtype=np.float64)
    m = 3
    k = -2
    list_of_inputs.append({"p": p, "m": m, "k": k})

    # Input 8: int32 1D array, order 4, larger constant
    p = np.array([2, 4, 6], dtype=np.int32)
    m = 4
    k = 10
    list_of_inputs.append({"p": p, "m": m, "k": k})

    # Input 9: Negative float32 array, order 1, negative constant
    p = np.array([-10.0, -20.0, -30.0], dtype=np.float32)
    m = 1
    k = -5
    list_of_inputs.append({"p": p, "m": m, "k": k})

    # Input 10: float64 array, order 2, large constant
    p = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
    m = 2
    k = 100
    list_of_inputs.append({"p": p, "m": m, "k": k})

    return list_of_inputs

generated_inputs["jax.numpy.polyint_1"] = polyint_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyint_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyint_1'.")


check_valid('jax.numpy.polyint', generated_inputs['jax.numpy.polyint_1'], lib="jax", suffix=1)
