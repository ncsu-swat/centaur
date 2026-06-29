
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fiedler_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard positive values
    a = np.array([1.0, 4.0, 12.0, 45.0, 77.0], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: 1D int32 array
    a = np.array([1, 4, 12, 45, 77], dtype=np.int32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: 1D float64 array with negative and positive values
    a = np.array([-10.5, 0.0, 5.5, -2.3, 100.1], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: 2D float32 array, batch of vectors
    a = np.random.randn(3, 5).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: 2D int32 array
    a = np.random.randint(-10, 10, size=(2, 4)).astype(np.int32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: 3D float32 array
    a = np.random.randn(2, 2, 3).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: 1D array with single element
    a = np.array([42.0], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: 1D array of zeros
    a = np.zeros(6, dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: 4D float64 array
    a = np.random.randn(2, 1, 3, 4).astype(np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: 1D array with large values
    a = np.array([1e5, -1e5, 0.0, 5e4], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    return list_of_inputs

generated_inputs["jax.scipy.linalg.fiedler"] = fiedler_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.fiedler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.fiedler'.")


check_valid('jax.scipy.linalg.fiedler', generated_inputs['jax.scipy.linalg.fiedler'], lib="jax", suffix=0)
