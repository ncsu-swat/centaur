
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_arrays_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    a = np.array(3.14, dtype=np.float32)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 2: 1D int32 array with positive and negative values
    a = np.array([-5, 0, 10, -20], dtype=np.int32)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 3: 2D float64 array
    a = np.array([[1.0, -2.5], [3.5, 4.0]], dtype=np.float64)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 4: 3D int64 array
    a = np.ones((2, 3, 4), dtype=np.int64)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 5: 4D float32 array
    a = np.random.randn(2, 1, 3, 4).astype(np.float32)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 6: 1D boolean array
    a = np.array([True, False, True, False], dtype=bool)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 7: 2D complex64 array
    a = np.array([[1 + 2j, -3j], [4 + 0j, 5 - 1j]], dtype=np.complex64)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 8: 0D int16 scalar with negative value
    a = np.array(-42, dtype=np.int16)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 9: 5D uint8 array
    a = np.zeros((1, 2, 1, 3, 1), dtype=np.uint8)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 10: 2D empty array (zero dimension)
    a = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    # Input 11: 1D float16 array
    a = np.array([-1.5, 2.5, -3.5], dtype=np.float16)
    list_of_inputs.append({"args": copy.deepcopy(a)})

    return list_of_inputs


generated_inputs["jax.numpy.broadcast_arrays"] = broadcast_arrays_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.broadcast_arrays' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.broadcast_arrays'.")


check_valid('jax.numpy.broadcast_arrays', generated_inputs['jax.numpy.broadcast_arrays'], lib="jax", suffix=0)
