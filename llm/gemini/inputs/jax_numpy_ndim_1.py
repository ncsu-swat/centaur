
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ndim_inputs():
    list_of_inputs = []

    # Input 1: 1D array of floats
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: 2D array of integers
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: 3D array of booleans
    a = np.ones((2, 2, 2), dtype=bool)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: 0D array (scalar)
    a = np.array(3.14, dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: 4D array of float16
    a = np.random.randn(2, 3, 1, 5).astype(np.float16)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: 5D array of int16
    a = np.random.randint(-10, 10, size=(1, 2, 1, 3, 1)).astype(np.int16)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: Empty 1D array
    a = np.array([], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: 2D complex array
    a = np.array([[1+2j, 3+4j]], dtype=np.complex128)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: 3D uint8 array
    a = np.random.randint(0, 255, size=(4, 4, 3)).astype(np.uint8)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: 6D empty-like array of int64
    a = np.zeros((1, 1, 1, 1, 1, 1), dtype=np.int64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    return list_of_inputs

generated_inputs["jax.numpy.ndim_1"] = ndim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ndim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ndim_1'.")


check_valid('jax.numpy.ndim', generated_inputs['jax.numpy.ndim_1'], lib="jax", suffix=1)
