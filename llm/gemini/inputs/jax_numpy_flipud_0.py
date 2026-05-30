
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def flipud_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    m = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 2: 2D int32 array (matrix)
    m = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 3: 3D bool array
    m = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=bool)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 4: 4D float64 array
    m = np.random.randn(2, 3, 4, 5).astype(np.float64)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 5: 1D int64 array with negative values
    m = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 6: 2D complex64 array
    m = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 7: 3D float16 array
    m = np.random.randn(2, 2, 2).astype(np.float16)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 8: 2D uint8 array
    m = np.arange(12, dtype=np.uint8).reshape(4, 3)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 9: 5D int16 array
    m = np.random.randint(-100, 100, size=(2, 2, 3, 3, 2)).astype(np.int16)
    list_of_inputs.append({"m": copy.deepcopy(m)})

    # Input 10: 1D array with single element
    m = np.array([42.0], dtype=np.float32)
    list_of_inputs.append({"m": copy.deepcopy(m)})

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
