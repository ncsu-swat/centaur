
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dstack_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    tup = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"tup": tup, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with negative values
    tup = np.array([-5, 0, 5, -10, 15], dtype=np.int32)
    input_dict = {"tup": tup, "dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array
    tup = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    input_dict = {"tup": tup, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array
    tup = np.random.randn(4, 5).astype(np.float64)
    input_dict = {"tup": tup, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int16 array
    tup = np.arange(12, dtype=np.int16).reshape(3, 4)
    input_dict = {"tup": tup, "dtype": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array
    tup = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"tup": tup, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int32 array with negative values
    tup = np.random.randint(-100, 100, size=(3, 2, 5), dtype=np.int32)
    input_dict = {"tup": tup, "dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D complex64 array
    tup = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(
        np.complex64
    )
    input_dict = {"tup": tup, "dtype": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D uint8 array
    tup = np.random.randint(0, 256, size=(2, 3, 2), dtype=np.uint8)
    input_dict = {"tup": tup, "dtype": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array
    tup = np.random.randn(2, 3, 4, 2).astype(np.float32)
    input_dict = {"tup": tup, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D boolean array
    tup = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    input_dict = {"tup": tup, "dtype": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 1D int64 array casting to float64
    tup = np.array([10, 20, 30, 40], dtype=np.int64)
    input_dict = {"tup": tup, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.dstack_1"] = dstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dstack_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dstack_1'.")


check_valid('jax.numpy.dstack', generated_inputs['jax.numpy.dstack_1'], lib="jax", suffix=1)
