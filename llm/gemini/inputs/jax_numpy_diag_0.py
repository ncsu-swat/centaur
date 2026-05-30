
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def diag_inputs():
    list_of_inputs = []

    # Input 1: 1-D array, float32, k=0
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    k = 0
    list_of_inputs.append({"v": v, "k": k})

    # Input 2: 1-D array, int32, k=2
    v = np.array([10, 20, 30], dtype=np.int32)
    k = 2
    list_of_inputs.append({"v": v, "k": k})

    # Input 3: 1-D array, float64, k=-1
    v = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    k = -1
    list_of_inputs.append({"v": v, "k": k})

    # Input 4: 2-D array (square), float32, k=0
    v = np.random.randn(4, 4).astype(np.float32)
    k = 0
    list_of_inputs.append({"v": v, "k": k})

    # Input 5: 2-D array (non-square), int32, k=1
    v = np.random.randint(0, 10, size=(3, 5)).astype(np.int32)
    k = 1
    list_of_inputs.append({"v": v, "k": k})

    # Input 6: 2-D array (non-square), float64, k=-2
    v = np.random.randn(5, 3).astype(np.float64)
    k = -2
    list_of_inputs.append({"v": v, "k": k})

    # Input 7: 1-D array, complex64, k=0
    v = np.array([1+2j, 3+4j], dtype=np.complex64)
    k = 0
    list_of_inputs.append({"v": v, "k": k})

    # Input 8: 2-D array, bool, k=1
    v = np.array([[True, False], [False, True]], dtype=np.bool_)
    k = 1
    list_of_inputs.append({"v": v, "k": k})

    # Input 9: 1-D array of size 1, int32, k=-3
    v = np.array([42], dtype=np.int32)
    k = -3
    list_of_inputs.append({"v": v, "k": k})

    # Input 10: 2-D array (large), float32, k=5
    v = np.random.randn(10, 10).astype(np.float32)
    k = 5
    list_of_inputs.append({"v": v, "k": k})

    # Input 11: 2-D array, float32, k=-5
    v = np.random.randn(8, 8).astype(np.float32)
    k = -5
    list_of_inputs.append({"v": v, "k": k})

    return list_of_inputs

generated_inputs["jax.numpy.diag"] = diag_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diag'.")


check_valid('jax.numpy.diag', generated_inputs['jax.numpy.diag'], lib="jax", suffix=0)
