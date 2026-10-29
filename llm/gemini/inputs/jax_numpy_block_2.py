
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def block_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    a1 = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arrays": a1}))

    # Input 2: 2D float32 array
    a2 = np.ones((2, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrays": a2}))

    # Input 3: 2D float32 array with negative values
    a3 = np.array([[-1.0, 2.5], [3.0, -4.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrays": a3}))

    # Input 4: 3D float32 array of zeros
    a4 = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrays": a4}))

    # Input 5: 4D float64 array
    a5 = np.ones((1, 2, 2, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arrays": a5}))

    # Input 6: 1D int64 array with negative values
    a6 = np.array([-10, 20, -30, 40], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"arrays": a6}))

    # Input 7: 2D boolean array
    a7 = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"arrays": a7}))

    # Input 8: 2D complex64 array
    a8 = np.array(
        [[1.0 + 2.0j, -3.0 + 0.0j], [0.0 - 1.0j, 4.0 + 5.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"arrays": a8}))

    # Input 9: 1D uint8 array
    a9 = np.array([0, 128, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"arrays": a9}))

    # Input 10: 2D float32 identity-like array
    a10 = np.eye(4, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrays": a10}))

    # Input 11: 3D int32 array with reshaped range
    a11 = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    list_of_inputs.append(copy.deepcopy({"arrays": a11}))

    return list_of_inputs


generated_inputs["jax.numpy.block_2"] = block_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.block_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.block_2'.")


check_valid('jax.numpy.block', generated_inputs['jax.numpy.block_2'], lib="jax", suffix=2)
