
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fliplr_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D integer array (2x2)
    m = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 2: 2D float array (3x4) with negative and positive values
    m = np.array([[-1.5, 2.0, -3.5, 4.0],
                  [5.5, -6.0, 7.5, -8.0],
                  [-9.5, 10.0, -11.5, 12.0]], dtype=np.float32)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 3: 3D float array (2x3x4)
    m = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 4: 4D integer array (2x2x3x3)
    m = np.random.randint(-10, 10, size=(2, 2, 3, 3)).astype(np.int64)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 5: 2D float64 array with larger dimensions (10x10)
    m = np.random.randn(10, 10).astype(np.float64)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 6: 2D array with 1 row (1x5)
    m = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 7: 2D array with 1 col (5x1)
    m = np.array([[1], [2], [3], [4], [5]], dtype=np.int32)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 8: 2D boolean array (3x3)
    m = np.array([[True, False, True],
                  [False, True, False],
                  [True, True, False]], dtype=bool)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 9: 3D complex array (2x4x3)
    m = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(np.complex64)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 10: 2D array with single element (2x2) of uint8
    m = np.arange(4, dtype=np.uint8).reshape(2, 2)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    return list_of_inputs

generated_inputs["jax.numpy.fliplr"] = fliplr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fliplr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fliplr'.")


check_valid('jax.numpy.fliplr', generated_inputs['jax.numpy.fliplr'], lib="jax", suffix=0)
