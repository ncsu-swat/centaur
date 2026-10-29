
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def diagflat_inputs():
    list_of_inputs = []

    # Input 1: 1D array with default-like k=0
    v = np.array([1, 2, 3, 4], dtype=np.int32)
    k = 0
    list_of_inputs.append({"v": v, "k": k})

    # Input 2: 1D float array with positive offset
    v = np.array([1.5, -2.5, 3.0], dtype=np.float32)
    k = 1
    list_of_inputs.append({"v": v, "k": k})

    # Input 3: 1D float array with negative offset
    v = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    k = -2
    list_of_inputs.append({"v": v, "k": k})

    # Input 4: 2D array flattened with k=0
    v = np.arange(6, dtype=np.int64).reshape(2, 3)
    k = 0
    list_of_inputs.append({"v": v, "k": k})

    # Input 5: 2D array flattened with positive offset
    v = np.array([[10, 20], [30, 40]], dtype=np.int32)
    k = 2
    list_of_inputs.append({"v": v, "k": k})

    # Input 6: 2D array flattened with negative offset
    v = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    k = -1
    list_of_inputs.append({"v": v, "k": k})

    # Input 7: 3D array flattened with k=0
    v = np.ones((2, 2, 2), dtype=np.float32)
    k = 0
    list_of_inputs.append({"v": v, "k": k})

    # Input 8: Boolean 1D array
    v = np.array([True, False, True], dtype=np.bool_)
    k = 0
    list_of_inputs.append({"v": v, "k": k})

    # Input 9: Complex 1D array with positive offset
    v = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    k = 3
    list_of_inputs.append({"v": v, "k": k})

    # Input 10: 0D scalar array with k=0
    v = np.array(42, dtype=np.int32)
    k = 0
    list_of_inputs.append({"v": v, "k": k})

    # Input 11: Empty 1D array
    v = np.array([], dtype=np.float32)
    k = 1
    list_of_inputs.append({"v": v, "k": k})

    # Input 12: Large negative offset
    v = np.array([5, 6], dtype=np.int32)
    k = -4
    list_of_inputs.append({"v": v, "k": k})

    return [copy.deepcopy(x) for x in list_of_inputs]

generated_inputs["jax.numpy.diagflat"] = diagflat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diagflat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diagflat'.")


check_valid('jax.numpy.diagflat', generated_inputs['jax.numpy.diagflat'], lib="jax", suffix=0)
