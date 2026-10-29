
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def empty_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 prototype, int32 dtype, shape 5
    prototype = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = np.dtype(np.int32)
    shape = 5
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    # Input 2: 2D int32 prototype, float64 dtype, shape 10
    prototype = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    dtype = np.dtype(np.float64)
    shape = 10
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    # Input 3: 3D float64 prototype, bool dtype, shape 1
    prototype = np.ones((2, 3, 4), dtype=np.float64)
    dtype = np.dtype(np.bool_)
    shape = 1
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    # Input 4: 0D scalar prototype, complex64 dtype, shape 4
    prototype = np.array(3.14, dtype=np.float32)
    dtype = np.dtype(np.complex64)
    shape = 4
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    # Input 5: 1D bool prototype, uint8 dtype, shape 0 (empty dimension)
    prototype = np.array([True, False, True], dtype=np.bool_)
    dtype = np.dtype(np.uint8)
    shape = 0
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    # Input 6: 4D uint8 prototype, int16 dtype, shape 8
    prototype = np.zeros((1, 2, 2, 1), dtype=np.uint8)
    dtype = np.dtype(np.int16)
    shape = 8
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    # Input 7: 2D complex64 prototype, float32 dtype, shape 16
    prototype = np.array([[1 + 2j, -2 - 1j], [0j, 3j]], dtype=np.complex64)
    dtype = np.dtype(np.float32)
    shape = 16
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    # Input 8: 1D int64 prototype with negative values, uint32 dtype, shape 3
    prototype = np.array([-100, -200, 300], dtype=np.int64)
    dtype = np.dtype(np.uint32)
    shape = 3
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    # Input 9: 2D float16 prototype, int8 dtype, shape 7
    prototype = np.random.randn(3, 3).astype(np.float16)
    dtype = np.dtype(np.int8)
    shape = 7
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    # Input 10: 3D int8 prototype, complex128 dtype, shape 12
    prototype = np.random.randint(-10, 10, size=(2, 2, 3), dtype=np.int8)
    dtype = np.dtype(np.complex128)
    shape = 12
    list_of_inputs.append(
        copy.deepcopy({"prototype": prototype, "dtype": dtype, "shape": shape})
    )

    return list_of_inputs


generated_inputs["jax.numpy.empty_like_2"] = empty_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_like_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_like_2'.")


check_valid('jax.numpy.empty_like', generated_inputs['jax.numpy.empty_like_2'], lib="jax", suffix=2)
