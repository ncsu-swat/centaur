
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def flipud_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    m = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 2: 2D float32 array with negative values
    m = np.array([[-1.5, 2.5], [3.0, -4.2], [0.0, 5.1]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 3: 2D integer array (square)
    m = np.arange(16, dtype=np.int64).reshape((4, 4))
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 4: 3D float64 array
    m = np.random.randn(3, 4, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 5: Boolean array
    m = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 6: 4D array of float16
    m = np.ones((2, 3, 2, 2), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 7: Complex array
    m = np.array(
        [[1.0 + 2.0j, -3.0 - 4.0j], [0.5 - 1.5j, 2.2 + 0.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 8: Single-element 1D array
    m = np.array([42.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 9: Empty array with at least 1 dimension
    m = np.empty((0, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 10: 5D uint8 array
    m = np.arange(32, dtype=np.uint8).reshape((2, 2, 2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 11: Array with dimension of length 1 along axis 0
    m = np.array([[10, 20, 30]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    return list_of_inputs


generated_inputs["jax.numpy.flipud"] = flipud_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flipud' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flipud'.")


check_valid('jax.numpy.flipud', generated_inputs['jax.numpy.flipud'], lib="jax", suffix=0)
